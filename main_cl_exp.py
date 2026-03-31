# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

import sys, argparse, os, copy
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score

import torch.distributed as dist
import random
import wandb
import warnings

from aeon.datasets import load_classification
from syntheticdataset import MixedSyntheticBagsConcatK
from utils import make_dirs, get_logger, maybe_mkdir_p
from mydataload import loadorean
from dba_dataloader import build_dba_for_timemil, build_dba_windows_for_mixed
from lookhead import Lookahead

from models.timemil_old import newTimeMIL, AmbiguousMIL
from models.millet import MILLET
from models.timemil import TimeMIL, originalTimeMIL
from models.expmil import AmbiguousMILwithCL
from compute_aopcr import compute_classwise_aopcr

from os.path import join

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
#   DDP utils
# ---------------------------------------------------------------------
def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        is_main = (rank == 0)
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        is_main = True
    return rank, world_size, local_rank, is_main


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------
#   Seed
# ---------------------------------------------------------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------
#   PrototypeBank
# ---------------------------------------------------------------------
class PrototypeBank:
    def __init__(self, num_classes: int, dim: int, device, momentum: float = 0.9):
        self.num_classes = num_classes
        self.dim = dim
        self.momentum = momentum
        self.device = device
        self.prototypes = torch.zeros(num_classes, dim, device=device)
        self.initialized = torch.zeros(num_classes, dtype=torch.bool, device=device)

    @torch.no_grad()
    def update(self, x_cls: torch.Tensor, bag_label: torch.Tensor):
        B, C, D = x_cls.shape
        for c in range(C):
            mask = bag_label[:, c] > 0
            if mask.any():
                proto_batch = x_cls[mask, c, :].mean(dim=0)
                if self.initialized[c]:
                    m = self.momentum
                    self.prototypes[c] = m * self.prototypes[c] + (1.0 - m) * proto_batch
                else:
                    self.prototypes[c] = proto_batch
                    self.initialized[c] = True

    def get(self):
        return self.prototypes, self.initialized

    @torch.no_grad()
    def sync(self, world_size: int):
        if (world_size <= 1) or (not dist.is_available()) or (not dist.is_initialized()):
            return
        dist.all_reduce(self.prototypes, op=dist.ReduceOp.SUM)
        self.prototypes /= float(world_size)
        init_float = self.initialized.to(self.prototypes.dtype)
        dist.all_reduce(init_float, op=dist.ReduceOp.SUM)
        self.initialized = init_float > 0.0


# ---------------------------------------------------------------------
#   Instance-Prototype Contrastive Loss
# ---------------------------------------------------------------------
def instance_prototype_contrastive_loss(
    x_seq: torch.Tensor,        # [B, T, D]
    bag_label: torch.Tensor,    # [B, C]
    proto_bank: PrototypeBank,
    tau: float = 0.1,
    sim_thresh: float = 0.7,
    win: int = 5,
):
    """
    Vectorized InfoNCE between instance embeddings and class prototypes.
    Direct non-ambiguous: max_c S[b,t,c] >= sim_thresh
    Ambiguous: use temporal neighbor window if direct fails.
    """
    device = x_seq.device
    prototypes, initialized = proto_bank.get()

    if not initialized.any():
        return torch.tensor(0.0, device=device)

    B, T, D = x_seq.shape
    C = prototypes.shape[0]

    x_norm = F.normalize(x_seq, dim=-1)
    p_norm = F.normalize(prototypes, dim=-1)

    S_full = torch.einsum('btd,cd->btc', x_norm, p_norm)
    S_valid = S_full.clone()

    valid_class_mask = (bag_label > 0).unsqueeze(1) & initialized.view(1, 1, C)
    S_valid = S_valid.masked_fill(~valid_class_mask, -1e9)

    S_tmax, c_argmax = S_valid.max(dim=-1)
    direct_pos_mask = (S_tmax >= sim_thresh)

    if win > 0:
        S_padded = F.pad(S_tmax, (win, win), value=-1e9)
        windows = S_padded.unfold(dimension=1, size=2 * win + 1, step=1)
        nei_max, idx_in_win = windows.max(dim=-1)
        neighbor_pos_mask = (~direct_pos_mask) & (nei_max >= sim_thresh)
        t_arange = torch.arange(T, device=device).view(1, T)
        neighbor_t = (t_arange - win + idx_in_win).clamp(0, T - 1)
        c_nei = c_argmax.gather(1, neighbor_t)
    else:
        neighbor_pos_mask = torch.zeros_like(S_tmax, dtype=torch.bool)
        c_nei = torch.zeros_like(c_argmax)

    anchors_mask = direct_pos_mask | neighbor_pos_mask
    if not anchors_mask.any():
        return torch.tensor(0.0, device=device)

    c_all = torch.where(direct_pos_mask, c_argmax, c_nei)

    b_idx = torch.arange(B, device=device).view(B, 1).expand(B, T)[anchors_mask]
    t_idx = torch.arange(T, device=device).view(1, T).expand(B, T)[anchors_mask]
    c_idx = c_all[anchors_mask]
    N = b_idx.size(0)

    z_anchor = x_norm[b_idx, t_idx, :]
    sim_all = torch.matmul(z_anchor, p_norm.t()) / tau
    pos_sim = sim_all[torch.arange(N, device=device), c_idx]
    log_all = torch.logsumexp(sim_all, dim=-1)
    base = pos_sim - log_all

    with torch.no_grad():
        counts = torch.bincount(c_idx, minlength=C).float()
        weights_per_class = torch.zeros(C, device=device)
        valid = counts > 0
        weights_per_class[valid] = counts.sum() / counts[valid]

    w = weights_per_class[c_idx]
    return -(base * w).sum() / (w.sum() + 1e-6)


# ---------------------------------------------------------------------
#   Train
# ---------------------------------------------------------------------
def train(trainloader, milnet, criterion, optimizer, epoch, args, device,
          proto_bank=None, is_main=True, world_size=1):
    milnet.train()
    sum_bag = 0.0
    sum_inst = 0.0
    sum_sparsity = 0.0
    sum_proto_inst = 0.0
    sum_total = 0.0
    n = 0

    for batch_id, (feats, label) in enumerate(trainloader):
        bag_feats = feats.to(device)
        bag_label = label.to(device)

        optimizer.zero_grad()

        if args.model == 'AmbiguousMIL':
            bag_prediction, instance_pred, weighted_instance_pred, non_weighted_instance_pred, \
                x_cls, x_seq, attn_layer1, attn_layer2 = milnet(bag_feats)
        elif args.model == 'MILLET':
            bag_prediction, instance_pred, interpretation = milnet(bag_feats)
            weighted_instance_pred = None
            x_cls, x_seq = None, None
        else:
            if epoch < args.epoch_des:
                bag_prediction, x_cls, attn_layer1, attn_layer2 = milnet(bag_feats, warmup=True)
            else:
                bag_prediction, x_cls, attn_layer1, attn_layer2 = milnet(bag_feats, warmup=False)
            instance_pred = None

        bag_loss = criterion(bag_prediction, bag_label)

        inst_loss = 0.0
        sparsity_loss = torch.tensor(0.0, device=device)
        proto_inst_loss = torch.tensor(0.0, device=device)

        if instance_pred is not None:
            inst_loss = criterion(instance_pred, bag_label)

            instance_pred_s = torch.sigmoid(weighted_instance_pred)
            sparsity_loss = instance_pred_s.mean(dim=1).mean()

            if (epoch >= args.epoch_des) and (proto_bank is not None):
                proto_bank.update(x_cls.detach(), bag_label)
                if world_size > 1:
                    proto_bank.sync(world_size)
                proto_inst_loss = instance_prototype_contrastive_loss(
                    x_seq, bag_label, proto_bank,
                    tau=args.proto_tau,
                    sim_thresh=args.proto_sim_thresh,
                    win=args.proto_win,
                )

        if args.model == 'AmbiguousMIL':
            loss = (
                args.bag_loss_w * bag_loss
                + args.inst_loss_w * inst_loss
                + args.sparsity_loss_w * sparsity_loss
                + args.proto_loss_w * proto_inst_loss
            )
        else:
            loss = bag_loss

        if is_main:
            sys.stdout.write(
                '\r [Train] Epoch %d | [%d/%d] bag loss: %.4f  total loss: %.4f' %
                (epoch, batch_id, len(trainloader), bag_loss.item(), loss.item())
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(milnet.parameters(), 2.0)
        optimizer.step()

        sum_bag += bag_loss.item()
        sum_inst += float(inst_loss) if isinstance(inst_loss, float) else float(inst_loss)
        sum_sparsity += sparsity_loss.item()
        sum_proto_inst += proto_inst_loss.item()
        sum_total += loss.item()
        n += 1

    if is_main and wandb.run is not None:
        wandb.log({
            "epoch": epoch,
            "train/bag_loss": sum_bag / max(1, n),
            "train/inst_loss": sum_inst / max(1, n),
            "train/sparsity_loss": sum_sparsity / max(1, n),
            "train/proto_inst_loss": sum_proto_inst / max(1, n),
            "train/total_loss": sum_total / max(1, n),
        }, step=epoch)

    return sum_total / max(1, n)


# ---------------------------------------------------------------------
#   Test / Validation
# ---------------------------------------------------------------------
def test(testloader, milnet, criterion, epoch, args, device,
         threshold: float = 0.5, proto_bank=None, is_main=True):
    model = milnet.module if isinstance(milnet, nn.parallel.DistributedDataParallel) else milnet
    model.eval()

    total_loss = 0.0
    all_labels, all_probs = [], []
    sum_bag = sum_inst = sum_sparsity = sum_proto_inst = sum_total = 0.0
    inst_total_correct = inst_total_count = 0
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

            if args.model == 'AmbiguousMIL':
                bag_prediction, instance_pred, weighted_instance_pred, non_weighted_instance_pred, \
                    x_cls, x_seq, attn_layer1, attn_layer2 = milnet(bag_feats)
            elif args.model == 'MILLET':
                bag_prediction, instance_pred, interpretation = model(bag_feats)
                weighted_instance_pred = None
                attn_layer2 = None
                x_seq = None
            elif args.model in ['newTimeMIL', 'TimeMIL']:
                out = model(bag_feats)
                instance_pred = None
                attn_layer2 = None
                bag_prediction = out[0] if isinstance(out, (tuple, list)) else out
                if isinstance(out, (tuple, list)):
                    attn_layer2 = out[3]
            else:
                raise ValueError(f"Unknown model: {args.model}")

            bag_loss = criterion(bag_prediction, bag_label)

            inst_loss = 0.0
            sparsity_loss = torch.tensor(0.0, device=device)
            proto_inst_loss = torch.tensor(0.0, device=device)

            if instance_pred is not None:
                inst_loss = criterion(instance_pred, bag_label)

                instance_pred_s = torch.sigmoid(weighted_instance_pred)
                sparsity_loss = instance_pred_s.mean(dim=1).mean()

                if (epoch >= args.epoch_des) and (proto_bank is not None):
                    proto_inst_loss = instance_prototype_contrastive_loss(
                        x_seq, bag_label, proto_bank,
                        tau=args.proto_tau,
                        sim_thresh=args.proto_sim_thresh,
                        win=args.proto_win,
                    )

            if args.datatype == "mixed" and y_inst is not None:
                y_inst = y_inst.to(device)
                y_inst_label = torch.argmax(y_inst, dim=2)

                if args.model == 'AmbiguousMIL':
                    pred_inst = torch.argmax(weighted_instance_pred, dim=2)
                elif args.model == 'MILLET' and instance_pred is not None:
                    pred_inst = torch.argmax(instance_pred, dim=2)
                elif args.model in ['newTimeMIL', 'TimeMIL'] and attn_layer2 is not None:
                    B, T, C = y_inst.shape
                    attn_cls = attn_layer2[:, :, :C, C:]
                    pred_inst = torch.argmax(attn_cls.mean(dim=1), dim=1)
                else:
                    pred_inst = None

                if pred_inst is not None:
                    inst_total_correct += (pred_inst == y_inst_label).sum().item()
                    inst_total_count += y_inst_label.numel()

            if args.model == 'AmbiguousMIL':
                loss = (
                    args.bag_loss_w * bag_loss
                    + args.inst_loss_w * inst_loss
                    + args.sparsity_loss_w * sparsity_loss
                    + args.proto_loss_w * proto_inst_loss
                )
            else:
                loss = bag_loss

            if is_main:
                sys.stdout.write(
                    '\r [Val]   Epoch %d | [%d/%d] bag loss: %.4f  total loss: %.4f' %
                    (epoch, batch_id, len(testloader), bag_loss.item(), loss.item())
                )

            total_loss += loss.item()

            if args.model == 'AmbiguousMIL':
                probs = torch.sigmoid(instance_pred).cpu().numpy()
            else:
                probs = torch.sigmoid(bag_prediction).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(label.cpu().numpy())

            sum_bag += bag_loss.item()
            sum_inst += float(inst_loss) if isinstance(inst_loss, float) else float(inst_loss)
            sum_sparsity += sparsity_loss.item()
            sum_proto_inst += proto_inst_loss.item()
            sum_total += loss.item()
            n += 1

    y_true = np.vstack(all_labels)
    y_prob = np.vstack(all_probs)
    y_pred = (y_prob >= threshold).astype(np.int32)

    inst_acc = float(inst_total_correct) / float(inst_total_count) if inst_total_count > 0 else None

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

    bag_acc = None
    if args.datatype == "original":
        true_cls = y_true.argmax(axis=1)
        pred_cls = y_prob.argmax(axis=1)
        bag_acc = float((true_cls == pred_cls).mean())

    if is_main and wandb.run is not None:
        log_dict = {
            "val/bag_loss": sum_bag / max(1, n),
            "val/inst_loss": sum_inst / max(1, n),
            "val/sparsity_loss": sum_sparsity / max(1, n),
            "val/proto_inst_loss": sum_proto_inst / max(1, n),
            "val/total_loss": sum_total / max(1, n),
        }
        if bag_acc is not None:
            log_dict["val/bag_acc"] = bag_acc
        wandb.log(log_dict, step=epoch)

    results = {
        "f1_micro": f1_micro, "f1_macro": f1_macro,
        "p_micro": p_micro,   "p_macro": p_macro,
        "r_micro": r_micro,   "r_macro": r_macro,
        "roc_auc_macro": roc_macro, "mAP_macro": ap_macro,
    }
    if bag_acc is not None:
        results["bag_acc"] = bag_acc
    if inst_acc is not None:
        results["inst_acc"] = inst_acc

    return total_loss / max(1, n), results


# ---------------------------------------------------------------------
#   Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Ambiguous MIL with Contrastive Learning for Time Series')
    parser.add_argument('--dataset', default="PenDigits", type=str)
    parser.add_argument('--datatype', default="mixed", type=str, help='original | mixed')
    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--feats_size', default=512, type=int)
    parser.add_argument('--lr', default=5e-3, type=float)
    parser.add_argument('--num_epochs', default=300, type=int)
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,))
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--dropout_patch', default=0.5, type=float)
    parser.add_argument('--dropout_node', default=0.2, type=float)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--model', default='AmbiguousMIL', type=str,
                        help='AmbiguousMIL | newTimeMIL | TimeMIL | MILLET')
    parser.add_argument('--prepared_npz', type=str, default='./data/PAMAP2.npz')
    parser.add_argument('--optimizer', default='adamw', type=str, help='adamw | sgd | adam')
    parser.add_argument('--save_dir', default='./savemodel/', type=str)
    parser.add_argument('--epoch_des', default=20, type=int, help='warmup epochs before enabling instance losses')
    parser.add_argument('--embed', default=128, type=int)
    parser.add_argument('--batchsize', default=64, type=int)

    # Prototype contrastive loss params
    parser.add_argument('--proto_tau', type=float, default=0.1)
    parser.add_argument('--proto_sim_thresh', type=float, default=0.5)
    parser.add_argument('--proto_win', type=int, default=5)
    parser.add_argument('--proto_momentum', type=float, default=0.9)

    # Loss weights
    parser.add_argument('--bag_loss_w', type=float, default=0.5)
    parser.add_argument('--inst_loss_w', type=float, default=0.2)
    parser.add_argument('--sparsity_loss_w', type=float, default=0.05)
    parser.add_argument('--proto_loss_w', type=float, default=0.2)

    # Data path
    parser.add_argument('--data_path', type=str, default='./data', help='Root path for dataset storage')

    # DBA dataset params
    parser.add_argument('--dba_root', type=str, default='./dba_data')
    parser.add_argument('--dba_window', type=int, default=12000)
    parser.add_argument('--dba_stride', type=int, default=6000)
    parser.add_argument('--dba_test_ratio', type=float, default=0.2)

    # MILLET params
    parser.add_argument('--millet_pooling', default='conjunctive', type=str)

    args = parser.parse_args()

    rank, world_size, local_rank, is_main = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    # ----- Directories -----
    args.save_dir = args.save_dir + 'InceptBackbone'
    dataset_root = join(args.save_dir, f'{args.dataset}')
    maybe_mkdir_p(dataset_root)

    if is_main:
        exp_path = make_dirs(dataset_root)
    else:
        exp_path = None

    if world_size > 1:
        exp_path_list = [exp_path]
        dist.broadcast_object_list(exp_path_list, src=0)
        exp_path = exp_path_list[0]
        dist.barrier()

    args.save_dir = exp_path
    maybe_mkdir_p(args.save_dir)
    version_name = os.path.basename(exp_path)

    if is_main:
        print(f'Dataset: {args.dataset} | Model: {args.model} | Run: {version_name}')
        wandb.init(project="TimeMIL", name=f"{args.dataset}_{args.model}_{version_name}", config=vars(args))
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("val/*",   step_metric="epoch")
        wandb.define_metric("score/*", step_metric="epoch")
        logger = get_logger(os.path.join(args.save_dir, 'Train_log.log'))

        option = vars(args)
        with open(os.path.join(args.save_dir, 'option.txt'), 'wt') as f:
            f.write('------------ Options -------------\n')
            for k, v in sorted(option.items()):
                f.write(f'{k}: {v}\n')
            f.write('-------------- End ----------------\n')
    else:
        logger = None

    criterion = nn.BCEWithLogitsLoss()

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

    # ----- Dataset -----
    if args.dataset in ['JapaneseVowels', 'SpokenArabicDigits', 'CharacterTrajectories', 'InsectWingbeat']:
        train_base = loadorean(args, split='train')
        test_base  = loadorean(args, split='test')
        base_T = train_base.max_len
        num_classes = train_base.num_class
        L_in = train_base.feat_in
        args.feats_size = L_in
        args.num_classes = num_classes

        if args.datatype == "original":
            trainset, testset = train_base, test_base
            seq_len = base_T
        elif args.datatype == "mixed":
            Xtr, ytr_idx = _dataset_to_X_yidx(train_base)
            Xte, yte_idx = _dataset_to_X_yidx(test_base)
            concat_k = 2
            trainset = MixedSyntheticBagsConcatK(X=Xtr, y_idx=ytr_idx, num_classes=num_classes,
                                                  total_bags=len(Xtr), concat_k=concat_k, seed=args.seed)
            testset  = MixedSyntheticBagsConcatK(X=Xte, y_idx=yte_idx, num_classes=num_classes,
                                                  total_bags=len(Xte), concat_k=concat_k, seed=args.seed + 1,
                                                  return_instance_labels=True)
            seq_len = concat_k * base_T
        else:
            raise ValueError(f"Unsupported datatype '{args.datatype}' for {args.dataset}")
        args.seq_len = seq_len

    elif args.dataset == 'PAMAP2':
        trainset = loadorean(args, split='train')
        testset  = loadorean(args, split='test')
        seq_len, num_classes, L_in = trainset.max_len, trainset.num_class, trainset.feat_in
        args.seq_len = seq_len
        args.feats_size = L_in
        args.num_classes = num_classes

    elif args.dataset == 'dba':
        if args.datatype == 'original':
            trainset, testset, seq_len, num_classes, L_in = build_dba_for_timemil(args)
        elif args.datatype == 'mixed':
            Xtr, ytr_idx, Xte, yte_idx, seq_len, num_classes, L_in = build_dba_windows_for_mixed(args)
            trainset = MixedSyntheticBagsConcatK(X=Xtr, y_idx=ytr_idx, num_classes=num_classes,
                                                  total_bags=len(Xtr), concat_k=2, seed=args.seed)
            testset  = MixedSyntheticBagsConcatK(X=Xte, y_idx=yte_idx, num_classes=num_classes,
                                                  total_bags=len(Xte), concat_k=2, seed=args.seed + 1,
                                                  return_instance_labels=True)
        else:
            raise ValueError(f"Unsupported datatype '{args.datatype}' for DBA")
        args.seq_len = seq_len
        args.feats_size = L_in
        args.num_classes = num_classes

    else:
        Xtr, ytr, meta = load_classification(name=args.dataset, split='train', extract_path=args.data_path)
        Xte, yte, _    = load_classification(name=args.dataset, split='test',  extract_path=args.data_path)

        word_to_idx = {cls: i for i, cls in enumerate(meta['class_values'])}
        ytr_idx = torch.tensor([word_to_idx[i] for i in ytr], dtype=torch.long)
        yte_idx = torch.tensor([word_to_idx[i] for i in yte], dtype=torch.long)

        Xtr = torch.from_numpy(Xtr).permute(0, 2, 1).float()
        Xte = torch.from_numpy(Xte).permute(0, 2, 1).float()

        num_classes = len(meta['class_values'])
        args.num_classes = num_classes
        L_in = Xtr.shape[-1]
        seq_len = max(21, Xte.shape[1])

        if args.datatype == 'mixed':
            trainset = MixedSyntheticBagsConcatK(X=Xtr, y_idx=ytr_idx, num_classes=num_classes,
                                                  total_bags=len(Xtr), seed=args.seed)
            testset  = MixedSyntheticBagsConcatK(X=Xte, y_idx=yte_idx, num_classes=num_classes,
                                                  total_bags=len(Xte), seed=args.seed + 1,
                                                  return_instance_labels=True)
        elif args.datatype == 'original':
            trainset = TensorDataset(Xtr, F.one_hot(ytr_idx, num_classes=num_classes).float())
            testset  = TensorDataset(Xte, F.one_hot(yte_idx, num_classes=num_classes).float())
        else:
            raise ValueError(f"Unsupported datatype '{args.datatype}'")

        args.seq_len = seq_len
        args.feats_size = L_in

    if is_main:
        print(f'num_classes={num_classes}, seq_len={seq_len}, feat_in={L_in}')

    # ----- Model -----
    if args.model == 'AmbiguousMIL':
        base_model = AmbiguousMILwithCL(args.feats_size, mDim=args.embed, n_classes=num_classes,
                                         dropout=args.dropout_node, is_instance=True).to(device)
        proto_bank = PrototypeBank(num_classes=num_classes, dim=args.embed, device=device, momentum=args.proto_momentum)
    elif args.model == 'newTimeMIL':
        base_model = TimeMIL(args.feats_size, mDim=args.embed, n_classes=num_classes,
                              dropout=args.dropout_node, max_seq_len=seq_len, is_instance=True).to(device)
        proto_bank = None
    elif args.model == 'TimeMIL':
        base_model = originalTimeMIL(args.feats_size, mDim=args.embed, n_classes=num_classes,
                                      dropout=args.dropout_node, max_seq_len=seq_len, is_instance=True).to(device)
        proto_bank = None
    elif args.model == 'MILLET':
        base_model = MILLET(args.feats_size, mDim=args.embed, n_classes=num_classes,
                             dropout=args.dropout_node, max_seq_len=seq_len,
                             pooling=args.millet_pooling, is_instance=True).to(device)
        proto_bank = None
    else:
        raise ValueError(f"Unknown model: {args.model}")

    milnet = (nn.parallel.DistributedDataParallel(
                  base_model,
                  device_ids=[local_rank] if device.type == "cuda" else None,
                  output_device=local_rank if device.type == "cuda" else None,
                  find_unused_parameters=True,
              ) if world_size > 1 else base_model)

    # ----- Optimizer -----
    if args.optimizer == 'adamw':
        optimizer = Lookahead(torch.optim.AdamW(milnet.parameters(), lr=args.lr, weight_decay=args.weight_decay))
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(milnet.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = Lookahead(torch.optim.Adam(milnet.parameters(), lr=args.lr, weight_decay=args.weight_decay))
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    # ----- Batch size -----
    batch_size_map = {
        frozenset(['DuckDuckGeese', 'PenDigits', 'FingerMovements', 'ERing', 'EigenWorms',
                   'HandMovementDirection', 'RacketSports', 'UWaveGestureLibrary']): 64,
        frozenset(['Heartbeat']): 32,
        frozenset(['EthanolConcentration', 'NATOPS', 'JapaneseVowels', 'SelfRegulationSCP1']): 16,
        frozenset(['PEMS-SF', 'SelfRegulationSCP2', 'AtrialFibrillation', 'Cricket']): 8,
        frozenset(['StandWalkJump', 'MotorImagery']): 1,
        frozenset(['Libras', 'Handwriting', 'Epilepsy', 'PhonemeSpectra']): 128,
        frozenset(['FaceDetection', 'LSST', 'ArticularyWordRecognition', 'BasicMotions']): 512,
    }
    batch = args.batchsize
    for ds_set, bs in batch_size_map.items():
        if args.dataset in ds_set:
            batch = bs
            break

    # ----- Dataloaders -----
    if world_size > 1:
        train_sampler = DistributedSampler(trainset, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        train_sampler = None

    trainloader = DataLoader(trainset, batch_size=batch, shuffle=(train_sampler is None),
                             num_workers=args.num_workers, drop_last=False, pin_memory=True,
                             sampler=train_sampler)
    testloader  = DataLoader(testset,  batch_size=batch, shuffle=False,
                             num_workers=args.num_workers, drop_last=False, pin_memory=True)

    def _four_sig(val):
        return float(f"{val:.4g}")

    def _is_better(new_res, best_res):
        primary_key = "bag_acc" if args.datatype == "original" else "mAP_macro"
        new_p  = new_res.get(primary_key, 0.0)
        best_p = best_res.get(primary_key, -float("inf"))
        if _four_sig(new_p) != _four_sig(best_p):
            return new_p > best_p
        new_f1  = new_res.get("f1_macro", 0.0)
        best_f1 = best_res.get("f1_macro", -float("inf"))
        if _four_sig(new_f1) != _four_sig(best_f1):
            return new_f1 > best_f1
        new_inst  = new_res.get("inst_acc")
        best_inst = best_res.get("inst_acc")
        if new_inst is not None and best_inst is not None:
            return _four_sig(new_inst) > _four_sig(best_inst)
        return False

    # ----- Training loop -----
    save_path = join(args.save_dir, 'weights')
    if is_main:
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(join(args.save_dir, 'lesion'), exist_ok=True)
    if world_size > 1:
        dist.barrier()

    results_best = None
    best_model_path = None

    for epoch in range(1, args.num_epochs + 1):
        if isinstance(trainloader.sampler, DistributedSampler):
            trainloader.sampler.set_epoch(epoch)

        train(trainloader, milnet, criterion, optimizer, epoch, args, device,
              proto_bank=proto_bank, is_main=is_main, world_size=world_size)

        if is_main:
            test_loss, results = test(testloader, milnet, criterion, epoch, args, device,
                                      threshold=0.5, proto_bank=proto_bank, is_main=is_main)

            if wandb.run is not None:
                log_score = {
                    "epoch": epoch,
                    "score/f1_micro": results["f1_micro"],
                    "score/f1_macro": results["f1_macro"],
                    "score/precision_micro": results["p_micro"],
                    "score/precision_macro": results["p_macro"],
                    "score/recall_micro": results["r_micro"],
                    "score/recall_macro": results["r_macro"],
                    "score/roc_auc_macro": results["roc_auc_macro"],
                    "score/mAP_macro": results["mAP_macro"],
                }
                if "bag_acc"  in results: log_score["score/acc"]       = results["bag_acc"]
                if "inst_acc" in results: log_score["score/inst_acc"]  = results["inst_acc"]
                wandb.log(log_score, step=epoch)

            if logger:
                metrics = (results["f1_micro"], results["f1_macro"],
                           results["p_micro"], results["p_macro"],
                           results["r_micro"], results["r_macro"],
                           results["roc_auc_macro"], results["mAP_macro"])
                msg = ('Epoch [%d/%d] test loss: %.4f | F1(mi)=%.4f F1(Ma)=%.4f '
                       'P(mi)=%.4f P(Ma)=%.4f R(mi)=%.4f R(Ma)=%.4f '
                       'ROC=%.4f mAP=%.4f') % ((epoch, args.num_epochs, test_loss) + metrics)
                if "bag_acc"  in results: msg += f" Acc={results['bag_acc']:.4f}"
                if "inst_acc" in results: msg += f" InstAcc={results['inst_acc']:.4f}"
                logger.info(msg)

            should_save = results_best is None or _is_better(results, results_best)
            if should_save and epoch >= args.epoch_des:
                results_best = copy.deepcopy(results)
                primary_key = "bag_acc" if args.datatype == "original" else "mAP_macro"
                print(f"  => Best {primary_key}: {results_best[primary_key]:.4f}")
                save_name = join(save_path, f'best_{args.model}.pth')
                best_model_path = save_name
                state_dict = milnet.module.state_dict() if isinstance(milnet, nn.parallel.DistributedDataParallel) else milnet.state_dict()
                torch.save(state_dict, save_name)
                if logger: logger.info(f'Best model saved: {save_name}')

    # ----- Best model logging -----
    if is_main and results_best is not None and logger:
        best = results_best
        msg = ('Best Results | F1(mi)=%.4f F1(Ma)=%.4f P(mi)=%.4f P(Ma)=%.4f '
               'R(mi)=%.4f R(Ma)=%.4f ROC=%.4f mAP=%.4f') % (
            best["f1_micro"], best["f1_macro"],
            best["p_micro"], best["p_macro"],
            best["r_micro"], best["r_macro"],
            best["roc_auc_macro"], best["mAP_macro"])
        if "bag_acc"  in best: msg += f" Acc={best['bag_acc']:.4f}"
        if "inst_acc" in best: msg += f" InstAcc={best['inst_acc']:.4f}"
        logger.info(msg)

    # ----- Final eval + AOPCR -----
    if is_main and best_model_path is not None:
        if logger: logger.info(f"Loading best model for final eval: {best_model_path}")

        model_map = {
            'AmbiguousMIL': lambda: AmbiguousMILwithCL(args.feats_size, mDim=args.embed,
                                                        n_classes=args.num_classes,
                                                        dropout=args.dropout_node, is_instance=True),
            'newTimeMIL':   lambda: TimeMIL(args.feats_size, mDim=args.embed, n_classes=args.num_classes,
                                            dropout=args.dropout_node, max_seq_len=args.seq_len, is_instance=True),
            'TimeMIL':      lambda: originalTimeMIL(args.feats_size, mDim=args.embed, n_classes=args.num_classes,
                                                    dropout=args.dropout_node, max_seq_len=args.seq_len, is_instance=True),
            'MILLET':       lambda: MILLET(args.feats_size, mDim=args.embed, n_classes=args.num_classes,
                                           dropout=args.dropout_node, max_seq_len=args.seq_len,
                                           pooling=args.millet_pooling, is_instance=True),
        }

        if args.model in model_map:
            eval_model = model_map[args.model]().to(device)
            eval_model.load_state_dict(torch.load(best_model_path, map_location=device))
            eval_model.eval()

            _, results_final = test(testloader, eval_model, criterion, epoch=args.num_epochs,
                                    args=args, device=device, threshold=0.5, is_main=is_main)
            if logger:
                msg = ('Final eval | F1(mi)=%.4f F1(Ma)=%.4f P(mi)=%.4f P(Ma)=%.4f '
                       'R(mi)=%.4f R(Ma)=%.4f ROC=%.4f mAP=%.4f') % (
                    results_final.get("f1_micro", 0), results_final.get("f1_macro", 0),
                    results_final.get("p_micro", 0),  results_final.get("p_macro", 0),
                    results_final.get("r_micro", 0),  results_final.get("r_macro", 0),
                    results_final.get("roc_auc_macro", 0), results_final.get("mAP_macro", 0))
                if "inst_acc" in results_final: msg += f" InstAcc={results_final['inst_acc']:.4f}"
                logger.info(msg)

            print("Computing AOPCR...")
            aopcr_loader = DataLoader(testset, batch_size=1, shuffle=False,
                                      num_workers=args.num_workers, pin_memory=True)
            aopcr_c, aopcr_w_avg, aopcr_mean, *_ = compute_classwise_aopcr(
                eval_model, aopcr_loader, args,
                stop=0.5, step=0.05, n_random=3, pred_threshold=0.5,
            )
            for c in range(args.num_classes):
                val = aopcr_c[c]
                msg = f"[AOPCR] class {c}: {'NaN' if np.isnan(val) else f'{val:.6f}'}"
                logger.info(msg) if logger else print(msg)
            summary = f"Weighted AOPCR: {aopcr_w_avg:.6f}, Mean AOPCR: {aopcr_mean:.6f}"
            logger.info(summary) if logger else print(summary)

    cleanup_distributed()


if __name__ == '__main__':
    main()
