# -*- coding: utf-8 -*-

"""
Evaluate a saved checkpoint on the test set.

Usage:
    python eval_checkpoint.py \
        --checkpoint ./savemodel/InceptBackbone/PenDigits/exp_0/weights/best_AmbiguousMIL.pth \
        --dataset PenDigits \
        --model AmbiguousMIL \
        --datatype mixed

    # with AOPCR computation
    python eval_checkpoint.py \
        --checkpoint ./savemodel/InceptBackbone/PenDigits/exp_0/weights/best_AmbiguousMIL.pth \
        --dataset PenDigits \
        --model AmbiguousMIL \
        --datatype mixed \
        --compute_aopcr
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import argparse, os, sys, random
import numpy as np
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score, average_precision_score)

from aeon.datasets import load_classification
from syntheticdataset import MixedSyntheticBagsConcatK
from mydataload import loadorean
from dba_dataloader import build_dba_for_timemil, build_dba_windows_for_mixed
from models.expmil import AmbiguousMILwithCL

import warnings
warnings.filterwarnings("ignore")

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def evaluate(testloader, milnet, args, device, threshold=0.5):
    milnet.eval()
    criterion = nn.BCEWithLogitsLoss()

    all_labels, all_probs = [], []
    inst_total_correct = inst_total_count = 0
    total_loss = 0.0
    n = 0

    with torch.no_grad():
        for batch_id, batch in enumerate(testloader):
            if args.datatype == "mixed":
                feats, label, y_inst = batch
            else:
                feats, label = batch
                y_inst = None

            bag_feats = feats.to(device)
            bag_label = label.to(device)

            bag_prediction, instance_pred, weighted_instance_pred, \
                non_weighted_instance_pred, x_cls, x_seq, attn_layer1, attn_layer2 = milnet(bag_feats)

            bag_loss = criterion(bag_prediction, bag_label)
            inst_loss = criterion(instance_pred, bag_label)
            instance_pred_s = torch.sigmoid(weighted_instance_pred)
            sparsity_loss = instance_pred_s.mean(dim=1).mean()

            loss = (args.bag_loss_w * bag_loss
                    + args.inst_loss_w * inst_loss
                    + args.sparsity_loss_w * sparsity_loss)
            total_loss += loss.item()

            # Instance accuracy (mixed only)
            if args.datatype == "mixed" and y_inst is not None:
                y_inst = y_inst.to(device)
                y_inst_label = torch.argmax(y_inst, dim=2)
                pred_inst = torch.argmax(weighted_instance_pred, dim=2)
                inst_total_correct += (pred_inst == y_inst_label).sum().item()
                inst_total_count += y_inst_label.numel()

            probs = torch.sigmoid(instance_pred).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(label.cpu().numpy())

            sys.stdout.write(f'\r  Evaluating [{batch_id + 1}/{len(testloader)}]')

            n += 1

    print()

    y_true = np.vstack(all_labels)
    y_prob = np.vstack(all_probs)
    y_pred = (y_prob >= threshold).astype(np.int32)

    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    p_micro  = precision_score(y_true, y_pred, average='micro', zero_division=0)
    p_macro  = precision_score(y_true, y_pred, average='macro', zero_division=0)
    r_micro  = recall_score(y_true, y_pred, average='micro', zero_division=0)
    r_macro  = recall_score(y_true, y_pred, average='macro', zero_division=0)

    roc_list, ap_list = [], []
    for c in range(y_true.shape[1]):
        if len(np.unique(y_true[:, c])) == 2:
            try:
                roc_list.append(roc_auc_score(y_true[:, c], y_prob[:, c]))
                ap_list.append(average_precision_score(y_true[:, c], y_prob[:, c]))
            except Exception:
                pass

    roc_macro = float(np.mean(roc_list)) if roc_list else 0.0
    ap_macro  = float(np.mean(ap_list))  if ap_list  else 0.0

    inst_acc = float(inst_total_correct) / float(inst_total_count) if inst_total_count > 0 else None

    bag_acc = None
    if args.datatype == "original":
        true_cls = y_true.argmax(axis=1)
        pred_cls = y_prob.argmax(axis=1)
        bag_acc = float((true_cls == pred_cls).mean())

    results = {
        "f1_micro": f1_micro, "f1_macro": f1_macro,
        "p_micro": p_micro,   "p_macro": p_macro,
        "r_micro": r_micro,   "r_macro": r_macro,
        "roc_auc_macro": roc_macro, "mAP_macro": ap_macro,
        "loss": total_loss / max(1, n),
    }
    if bag_acc is not None:
        results["bag_acc"] = bag_acc
    if inst_acc is not None:
        results["inst_acc"] = inst_acc

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate a saved checkpoint')
    parser.add_argument('--checkpoint', required=True, type=str, help='Path to .pth checkpoint')
    parser.add_argument('--dataset', default="PenDigits", type=str)
    parser.add_argument('--datatype', default="mixed", type=str, help='original | mixed')
    parser.add_argument('--model', default='AmbiguousMIL', type=str)
    parser.add_argument('--embed', default=128, type=int)
    parser.add_argument('--dropout_node', default=0.2, type=float)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--batchsize', default=64, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--compute_aopcr', action='store_true', help='Compute AOPCR interpretability metric')
    parser.add_argument('--data_path', type=str, default='./data', help='Root path for dataset storage')
    parser.add_argument('--prepared_npz', type=str, default='./data/PAMAP2.npz')
    parser.add_argument('--dba_root', type=str, default='./dba_data')
    parser.add_argument('--dba_window', type=int, default=12000)
    parser.add_argument('--dba_stride', type=int, default=6000)
    parser.add_argument('--dba_test_ratio', type=float, default=0.2)

    # Loss weights (for loss reporting only)
    parser.add_argument('--bag_loss_w', type=float, default=0.5)
    parser.add_argument('--inst_loss_w', type=float, default=0.2)
    parser.add_argument('--sparsity_loss_w', type=float, default=0.05)
    parser.add_argument('--proto_loss_w', type=float, default=0.2)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # ----- Dataset -----
    def _dataset_to_X_yidx(ds):
        X_list, y_list = [], []
        for i in range(len(ds)):
            item = ds[i]
            x, y = item[0], item[1]
            if not torch.is_tensor(x):
                x = torch.tensor(x)
            if not torch.is_tensor(y):
                y = torch.tensor(y)
            y_idx = int(y.argmax().item()) if y.ndim > 0 and y.numel() > 1 else int(y.item())
            X_list.append(x.float())
            y_list.append(y_idx)
        X = torch.stack(X_list, dim=0)
        y_idx = torch.tensor(y_list, dtype=torch.long)
        return X, y_idx

    if args.dataset in ['JapaneseVowels', 'SpokenArabicDigits', 'CharacterTrajectories', 'InsectWingbeat']:
        test_base = loadorean(args, split='test')
        train_base = loadorean(args, split='train')
        base_T = train_base.max_len
        num_classes = test_base.num_class
        L_in = test_base.feat_in
        args.feats_size = L_in
        args.num_classes = num_classes

        if args.datatype == "original":
            testset = test_base
            seq_len = base_T
        elif args.datatype == "mixed":
            Xte, yte_idx = _dataset_to_X_yidx(test_base)
            concat_k = 2
            testset = MixedSyntheticBagsConcatK(X=Xte, y_idx=yte_idx, num_classes=num_classes,
                                                  total_bags=len(Xte), concat_k=concat_k, seed=args.seed + 1,
                                                  return_instance_labels=True)
            seq_len = concat_k * base_T

    elif args.dataset == 'PAMAP2':
        testset = loadorean(args, split='test')
        seq_len, num_classes, L_in = testset.max_len, testset.num_class, testset.feat_in
        args.feats_size = L_in
        args.num_classes = num_classes

    elif args.dataset == 'dba':
        if args.datatype == 'original':
            _, testset, seq_len, num_classes, L_in = build_dba_for_timemil(args)
        elif args.datatype == 'mixed':
            _, _, Xte, yte_idx, seq_len, num_classes, L_in = build_dba_windows_for_mixed(args)
            testset = MixedSyntheticBagsConcatK(X=Xte, y_idx=yte_idx, num_classes=num_classes,
                                                  total_bags=len(Xte), concat_k=2, seed=args.seed + 1,
                                                  return_instance_labels=True)

    else:
        Xtr, ytr, meta = load_classification(name=args.dataset, split='train', extract_path=args.data_path)
        Xte, yte, _ = load_classification(name=args.dataset, split='test', extract_path=args.data_path)

        word_to_idx = {cls: i for i, cls in enumerate(meta['class_values'])}
        yte_idx = torch.tensor([word_to_idx[i] for i in yte], dtype=torch.long)
        Xte = torch.from_numpy(Xte).permute(0, 2, 1).float()

        num_classes = len(meta['class_values'])
        L_in = Xte.shape[-1]
        seq_len = max(21, Xte.shape[1])

        if args.datatype == 'mixed':
            testset = MixedSyntheticBagsConcatK(X=Xte, y_idx=yte_idx, num_classes=num_classes,
                                                  total_bags=len(Xte), seed=args.seed + 1,
                                                  return_instance_labels=True)
        elif args.datatype == 'original':
            testset = TensorDataset(Xte, F.one_hot(yte_idx, num_classes=num_classes).float())

    args.num_classes = num_classes
    args.feats_size = L_in
    args.seq_len = seq_len

    print(f'Dataset: {args.dataset} | Model: {args.model} | datatype: {args.datatype}')
    print(f'num_classes={num_classes}, seq_len={seq_len}, feat_in={L_in}')

    # ----- Model -----
    model = AmbiguousMILwithCL(
        args.feats_size, mDim=args.embed, n_classes=num_classes,
        dropout=args.dropout_node, is_instance=True
    ).to(device)

    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    print(f'Loaded checkpoint: {args.checkpoint}')

    # ----- Dataloader -----
    testloader = DataLoader(testset, batch_size=args.batchsize, shuffle=False,
                            num_workers=args.num_workers, drop_last=False, pin_memory=True)

    # ----- Evaluate -----
    print('\n--- Evaluation Results ---')
    results = evaluate(testloader, model, args, device)

    print(f'  F1 (micro):  {results["f1_micro"]:.4f}')
    print(f'  F1 (macro):  {results["f1_macro"]:.4f}')
    print(f'  Prec (micro): {results["p_micro"]:.4f}')
    print(f'  Prec (macro): {results["p_macro"]:.4f}')
    print(f'  Recall (micro): {results["r_micro"]:.4f}')
    print(f'  Recall (macro): {results["r_macro"]:.4f}')
    print(f'  ROC-AUC:     {results["roc_auc_macro"]:.4f}')
    print(f'  mAP (macro): {results["mAP_macro"]:.4f}')
    print(f'  Loss:        {results["loss"]:.4f}')
    if "bag_acc" in results:
        print(f'  Bag Acc:     {results["bag_acc"]:.4f}')
    if "inst_acc" in results:
        print(f'  Inst Acc:    {results["inst_acc"]:.4f}')

    # ----- AOPCR -----
    if args.compute_aopcr:
        from compute_aopcr import compute_classwise_aopcr
        print('\n--- Computing AOPCR ---')
        aopcr_loader = DataLoader(testset, batch_size=1, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=True)
        aopcr_c, aopcr_w_avg, aopcr_mean, aopcr_overall, _, _, _, counts = compute_classwise_aopcr(
            model, aopcr_loader, args,
            stop=0.5, step=0.05, n_random=3, pred_threshold=0.5,
        )
        print()
        for c in range(num_classes):
            val = aopcr_c[c]
            cnt = int(counts[c])
            print(f'  Class {c}: AOPCR={val:.6f}  (n={cnt})')
        print(f'  Weighted AOPCR: {aopcr_w_avg:.6f}')
        print(f'  Mean AOPCR:     {aopcr_mean:.6f}')
        if aopcr_overall is not None:
            print(f'  Overall AOPCR:  {aopcr_overall:.6f}')

    print('\nDone.')


if __name__ == '__main__':
    main()
