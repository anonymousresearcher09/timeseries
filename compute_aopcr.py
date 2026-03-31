# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

import sys, argparse, os, random
import numpy as np
import warnings

from sklearn.metrics import roc_auc_score, average_precision_score

from aeon.datasets import load_classification
from syntheticdataset import *
from utils import *
from mydataload import loadorean

from models.expmil import AmbiguousMILwithCL

warnings.filterwarnings("ignore")

# ------------------------ Seed ------------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def extract_attn_importance(attn_layer2: torch.Tensor,
                            T: int,
                            num_classes: int,
                            target_c: int,
                            model_name: str,
                            args) -> torch.Tensor:
    """
    Common function for TimeMIL / newTimeMIL / AmbiguousMIL
    Extracts importance score[t] based on 'class-token → time-token' attention.

    attn_layer2: [B, H, L, L] or [B, L, L]
    T: sequence length
    num_classes: number of classes
    target_c: target class index for importance
    model_name: 'TimeMIL' / 'newTimeMIL' / 'AmbiguousMIL'
    return: [B, T]  (per-timestep importance)
    """
    # 1) Average over heads to get [B, L, L]
    if attn_layer2.dim() == 4:
        # [B, H, L, L] -> average over heads
        attn = attn_layer2.mean(dim=1)     # [B, L, L]
    elif attn_layer2.dim() == 3:
        attn = attn_layer2                 # [B, L, L]
    else:
        raise ValueError(f"Unexpected attn_layer2 shape: {attn_layer2.shape}")

    B, L, K = attn.shape
    if K != L:
        raise ValueError(f"Attention is not square: {attn.shape}")

    C = args.num_classes
    cls_idx = target_c
    if model_name in ['newTimeMIL','AmbiguousMIL']:
        # query = class tokens (0..C_att-1), key = time tokens (C_att..C_att+T-1)
        attn_cls2time = attn[:, :C, C:]     # [B, C_att, T]

        # Select only the class token for the target class
        scores = attn_cls2time[:, cls_idx, :]              # [B, T]
    elif model_name == 'TimeMIL': # Single class token
        # query = class token (0), key = time tokens (1..T)
        attn_cls2time = attn[:, 0:1, 1:]    # [B, 1, T]
        scores = attn_cls2time[:, 0, :]                     # [B, T]
    return scores


# ------------------------------------------------------
#  Core AOPCR computation function
# ------------------------------------------------------
@torch.no_grad()
def compute_classwise_aopcr(
    milnet,
    testloader,
    args,
    stop: float = 0.5,
    step: float = 0.05,
    n_random: int = 3,
    pred_threshold: float = 0.5,
):
    """
    Computes 'class-wise AOPCR' for TimeMIL / newTimeMIL / AmbiguousMIL
    (supports multi-label).

    - Instances are not removed, only zero-masked
    - attn_layer2 is parsed the same way as in the test code
    - For each bag, computes curves for (1) classes with label >= 0.5,
      or if none, (2) the predicted argmax class
    """
    device = next(milnet.parameters()).device
    num_classes = args.num_classes

    # Perturbation ratios (0% to stop in step increments)
    # alphas[0] = 0 (no perturb), alphas[1:] = actual perturbation
    alphas = torch.arange(0.0, stop + 1e-8, step, device=device)
    n_steps = len(alphas)

    # Class-wise statistics
    aopcr_per_class = torch.zeros(num_classes, device=device)
    counts = torch.zeros(num_classes, device=device)          # per-class bag count
    M_expl = torch.zeros(num_classes, n_steps, device=device) # per-class explanation curve mean
    M_rand = torch.zeros(num_classes, n_steps, device=device) # per-class random curve mean

    milnet.eval()

    total_aopcr_sum = 0.0

    for batch in testloader:
        # testloader: assumes (feats, label, y_inst) structure (same as current code)

        if len(batch) == 3:
            feats, bag_label, y_inst = batch
        else:
            feats, bag_label = batch
            y_inst = None

        x = feats.to(device)          # [B, T, D]
        y_bag = bag_label.to(device)  # [B, C] (multi-hot)
        x = x.contiguous()
        batch_size, T, D = x.shape

        # ----- Model forward (original) -----
        if args.model == 'AmbiguousMIL':
            out = milnet(x)
            if not isinstance(out, (tuple, list)):
                raise ValueError("AmbiguousMIL output must be a tuple/list")
            prototype_logits, instance_pred, weighted_instance_pred, non_weighted_instance_pred, x_cls, x_seq, attn_layer1, attn_layer2 = out
            logits = instance_pred
            prob = torch.sigmoid(instance_pred)  # bag-level prob from instance logits
            instance_logits = weighted_instance_pred
        elif args.model == 'TimeMIL':
            out = milnet(x)
            logits, x_cls, attn_layer1, attn_layer2 = out
            prob = torch.sigmoid(logits)
            instance_logits = None
        elif args.model == 'newTimeMIL':
            out = milnet(x)
            logits, x_cls, attn_layer1, attn_layer2 = out
            prob = torch.sigmoid(logits)
            instance_logits = None
        else:
            raise ValueError(f"Unknown model name: {args.model}")

        for b in range(batch_size):
            if y_bag.dim() == 1:
                y_row = y_bag
            else:
                y_row = y_bag[b]

            if args.datatype == 'original':
                # For original dataset (single-label)
                target_classes = torch.tensor([prob[b].argmax()], device=device)
            else:
                # For mixed / synthetic dataset (multi-label)
                target_classes = (prob[b] > pred_threshold).nonzero(as_tuple=False).flatten()
            if target_classes.numel() == 0:
                continue

            for cls_tensor in target_classes:
                pred_c = int(cls_tensor.item())

                # ----- Compute timestep importance scores -----
                if args.model == 'AmbiguousMIL':
                    # instance_logits: [B, T, C] -> use target class prob after softmax
                    # s_all = torch.softmax(instance_logits[b], dim=-1)   # [T, C]
                    s_all = instance_logits[b]   # [T, C]
                    scores = s_all[:, pred_c]                           # [T]
                else:
                    # TimeMIL / newTimeMIL: based on class-token → time-token attention
                    scores = extract_attn_importance(
                        attn_layer2=attn_layer2,
                        T=T,
                        num_classes=num_classes,
                        target_c=pred_c,
                        model_name=args.model,
                        args=args
                    )[b]                                                # [T]

                scores = scores.detach()

                # Scores length may differ from input T (due to model internal processing)
                T_scores = scores.size(0)
                if T_scores != T:
                    # Interpolate or truncate if length differs
                    if T_scores < T:
                        # Interpolate scores to length T
                        scores = torch.nn.functional.interpolate(
                            scores.view(1, 1, -1), size=T, mode='linear', align_corners=False
                        ).view(-1)
                    else:
                        # Truncate scores to length T
                        scores = scores[:T]

                # Timestep indices sorted by importance (descending)
                sorted_idx = torch.argsort(scores, dim=0, descending=True)  # [T]

                # Original logit (before perturbation)
                orig_logit = logits[b, pred_c].item()

                # perturbation curves
                curve_expl = torch.zeros(n_steps, device=device)
                curve_expl[0] = orig_logit

                curves_rand = torch.zeros(n_random, n_steps, device=device)
                curves_rand[:, 0] = orig_logit

                # ----- perturbation loop -----
                for step_i, alpha in enumerate(alphas[1:], start=1):
                    # Zero-mask timesteps by alpha ratio
                    k = int(round(alpha.item() * T))
                    k = min(max(k, 1), T)  # always in range [1, T]

                    # ---- Explanation-based (expl) ----
                    idx_remove_expl = sorted_idx[:k]           # [k], guaranteed 0 <= idx < T
                    x_pert_expl = x[b:b+1].clone()
                    x_pert_expl[:, idx_remove_expl, :] = 0.0   # keep length, zero-mask important timesteps

                    out_expl = milnet(x_pert_expl)
                    if isinstance(out_expl, tuple):
                        if args.model == 'AmbiguousMIL':
                            logits_expl = out_expl[1]    # (prototype_logits, instance_pred, ...)
                        else:
                            logits_expl = out_expl[0]              # (logits, ...) format
                    else:
                        logits_expl = out_expl
                    curve_expl[step_i] = logits_expl[0, pred_c].item()

                    # ---- Random-based (rand) ----
                    for r in range(n_random):
                        rand_perm = torch.randperm(T, device=device)
                        idx_remove_rand = rand_perm[:k]
                        x_pert_rand = x[b:b+1].clone()
                        x_pert_rand[:, idx_remove_rand, :] = 0.0

                        out_rand = milnet(x_pert_rand)
                        if isinstance(out_rand, tuple):
                            if args.model == 'AmbiguousMIL':
                                logits_rand = out_rand[1]    # (prototype_logits, instance_pred, ...)
                            else:
                                logits_rand = out_rand[0]              # (logits, ...) format
                        else:
                            logits_rand = out_rand
                        curves_rand[r, step_i] = logits_rand[0, pred_c].item()

                # ----- Convert to logit drop curves -----
                drop_expl = orig_logit - curve_expl                       # [n_steps]
                drop_rand = orig_logit - curves_rand.mean(dim=0)          # [n_steps]

                # AOPC = mean over steps (simple average assuming uniform step intervals)
                aopc_expl = drop_expl.mean().item()
                aopc_rand = drop_rand.mean().item()
                aopcr = aopc_expl - aopc_rand
                if args.datatype == 'original':
                    total_aopcr_sum += aopcr

                # Accumulate class-wise statistics
                aopcr_per_class[pred_c] += aopcr
                counts[pred_c] += 1
                M_expl[pred_c] += drop_expl
                M_rand[pred_c] += drop_rand

    # ----- Per-class average and overall summary -----
    valid = counts > 0
    aopcr_per_class[valid] /= counts[valid]                  # per-class average
    M_expl[valid] /= counts[valid].unsqueeze(1)
    M_rand[valid] /= counts[valid].unsqueeze(1)

    # Weighted average (weighted by bag count)
    total = counts.sum()
    if total > 0:
        weights = counts / total
        aopcr_weighted = (aopcr_per_class * weights).sum().item()
    else:
        aopcr_weighted = 0.0

    # Simple average (valid classes only)
    if valid.any():
        aopcr_mean = aopcr_per_class[valid].mean().item()
    else:
        aopcr_mean = 0.0

    if args.datatype == 'original':
        total_bags = counts.sum().item()
        aopcr_overall_mean = total_aopcr_sum / total_bags

    return (
        aopcr_per_class.cpu().numpy(),  # per-class AOPCR (C,)
        aopcr_weighted,                 # weighted average AOPCR (scalar)
        aopcr_mean,                     # simple mean over classes (scalar)
        aopcr_overall_mean if args.datatype == 'original' else None,  # Overall AOPCR sum (only for original dataset)
        M_expl.cpu().numpy(),           # class-wise explanation curves (C, n_steps)
        M_rand.cpu().numpy(),           # class-wise random curves (C, n_steps)
        alphas.cpu().numpy(),           # perturbation ratios (n_steps,)
        counts.cpu().numpy(),           # per-class bag counts (C,)
    )
