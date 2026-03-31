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
    TimeMIL / newTimeMIL / AmbiguousMIL 공통으로
    'class-token → time-token' attention 기반 중요도 score[t]를 뽑는 함수.

    attn_layer2: [B, H, L, L] 또는 [B, L, L]
    T: 시퀀스 길이
    num_classes: 클래스 개수
    target_c: 중요도를 볼 target class index
    model_name: 'TimeMIL' / 'newTimeMIL' / 'AmbiguousMIL'
    return: [B, T]  (각 timestep 중요도)
    """
    # 1) head 평균해서 [B, L, L]로 통일
    if attn_layer2.dim() == 4:
        # [B, H, L, L] -> head 평균
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

        # target class에 해당하는 class token만 선택
        scores = attn_cls2time[:, cls_idx, :]              # [B, T]
    elif model_name == 'TimeMIL': # Single class token
        # query = class token (0), key = time tokens (1..T)
        attn_cls2time = attn[:, 0:1, 1:]    # [B, 1, T]
        scores = attn_cls2time[:, 0, :]                     # [B, T]
    return scores


# ------------------------------------------------------
#  AOPCR 계산 핵심 함수
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
    TimeMIL / newTimeMIL / AmbiguousMIL 에 대해
    'class-wise AOPCR'을 계산하는 함수 (multi-label 대응).

    - instance는 제거하지 않고 0으로 마스킹만 함
    - attn_layer2는 test 코드와 동일한 방식으로 파싱
    - 각 bag에서 (1) label이 0.5 이상인 클래스들, 없으면
      (2) 예측 argmax 클래스를 대상으로 curve를 계산
    """
    device = next(milnet.parameters()).device
    num_classes = args.num_classes

    # perturbation 비율 (0% ~ stop까지 step 간격)
    # alphas[0] = 0 (no perturb), alphas[1:] = 실제 perturb
    alphas = torch.arange(0.0, stop + 1e-8, step, device=device)
    n_steps = len(alphas)

    # class-wise 통계
    aopcr_per_class = torch.zeros(num_classes, device=device)
    counts = torch.zeros(num_classes, device=device)          # class별 bag 개수
    M_expl = torch.zeros(num_classes, n_steps, device=device) # class별 explanation curve 평균
    M_rand = torch.zeros(num_classes, n_steps, device=device) # class별 random curve 평균

    milnet.eval()

    total_aopcr_sum = 0.0

    for batch in testloader:
        # testloader: (feats, label, y_inst) 구조라고 가정 (지금 코드랑 동일)

        if len(batch) == 3:
            feats, bag_label, y_inst = batch
        else:
            feats, bag_label = batch
            y_inst = None

        x = feats.to(device)          # [B, T, D]
        y_bag = bag_label.to(device)  # [B, C] (multi-hot)
        x = x.contiguous()
        batch_size, T, D = x.shape

        # ----- 모델 forward (원본) -----
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
                # original 데이터셋인 경우 (single-label)
                target_classes = torch.tensor([prob[b].argmax()], device=device)
            else:
                # mixed / synthetic 데이터셋인 경우 (multi-label)
                target_classes = (prob[b] > pred_threshold).nonzero(as_tuple=False).flatten()
            if target_classes.numel() == 0:
                continue

            for cls_tensor in target_classes:
                pred_c = int(cls_tensor.item())

                # ----- timestep 중요도 score 계산 -----
                if args.model == 'AmbiguousMIL':
                    # instance_logits: [B, T, C] -> softmax 후 target class prob 사용
                    # s_all = torch.softmax(instance_logits[b], dim=-1)   # [T, C]
                    s_all = instance_logits[b]   # [T, C]
                    scores = s_all[:, pred_c]                           # [T]
                else:
                    # TimeMIL / newTimeMIL: class-token → time-token attention 기반
                    scores = extract_attn_importance(
                        attn_layer2=attn_layer2,
                        T=T,
                        num_classes=num_classes,
                        target_c=pred_c,
                        model_name=args.model,
                        args=args
                    )[b]                                                # [T]

                scores = scores.detach()

                # scores 길이가 입력 T와 다를 수 있음 (모델 내부 처리로 인해)
                T_scores = scores.size(0)
                if T_scores != T:
                    # 길이가 다르면 interpolate하거나 truncate
                    if T_scores < T:
                        # scores를 T 길이로 보간
                        scores = torch.nn.functional.interpolate(
                            scores.view(1, 1, -1), size=T, mode='linear', align_corners=False
                        ).view(-1)
                    else:
                        # scores를 T 길이로 자름
                        scores = scores[:T]

                # 중요도 큰 순으로 정렬된 timestep index
                sorted_idx = torch.argsort(scores, dim=0, descending=True)  # [T]

                # 원본 logit (perturb 전)
                orig_logit = logits[b, pred_c].item()

                # perturbation curves
                curve_expl = torch.zeros(n_steps, device=device)
                curve_expl[0] = orig_logit

                curves_rand = torch.zeros(n_random, n_steps, device=device)
                curves_rand[:, 0] = orig_logit

                # ----- perturbation loop -----
                for step_i, alpha in enumerate(alphas[1:], start=1):
                    # alpha 비율만큼 timestep을 0으로 마스킹
                    k = int(round(alpha.item() * T))
                    k = min(max(k, 1), T)  # 항상 1~T 범위

                    # ---- 설명 기반 (expl) ----
                    idx_remove_expl = sorted_idx[:k]           # [k], 0 <= idx < T 보장
                    x_pert_expl = x[b:b+1].clone()
                    x_pert_expl[:, idx_remove_expl, :] = 0.0   # 길이는 그대로, 중요한 timestep만 0

                    out_expl = milnet(x_pert_expl)
                    if isinstance(out_expl, tuple):
                        if args.model == 'AmbiguousMIL':
                            logits_expl = out_expl[1]    # (prototype_logits, instance_pred, ...)
                        else:
                            logits_expl = out_expl[0]              # (logits, ...) 형태
                    else:
                        logits_expl = out_expl
                    curve_expl[step_i] = logits_expl[0, pred_c].item()

                    # ---- 랜덤 기반 (rand) ----
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
                                logits_rand = out_rand[0]              # (logits, ...) 형태
                        else:
                            logits_rand = out_rand
                        curves_rand[r, step_i] = logits_rand[0, pred_c].item()

                # ----- logit drop 기준 curve로 변환 -----
                drop_expl = orig_logit - curve_expl                       # [n_steps]
                drop_rand = orig_logit - curves_rand.mean(dim=0)          # [n_steps]

                # AOPC = mean over steps (step 간격을 균일하다고 보고 단순 평균)
                aopc_expl = drop_expl.mean().item()
                aopc_rand = drop_rand.mean().item()
                aopcr = aopc_expl - aopc_rand
                if args.datatype == 'original':
                    total_aopcr_sum += aopcr

                # class-wise 통계 누적
                aopcr_per_class[pred_c] += aopcr
                counts[pred_c] += 1
                M_expl[pred_c] += drop_expl
                M_rand[pred_c] += drop_rand

    # ----- 클래스별 평균 및 전체 요약 -----
    valid = counts > 0
    aopcr_per_class[valid] /= counts[valid]                  # per-class 평균
    M_expl[valid] /= counts[valid].unsqueeze(1)
    M_rand[valid] /= counts[valid].unsqueeze(1)

    # weighted 평균 (bag 수로 가중)
    total = counts.sum()
    if total > 0:
        weights = counts / total
        aopcr_weighted = (aopcr_per_class * weights).sum().item()
    else:
        aopcr_weighted = 0.0

    # 단순 평균 (valid class만)
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
        aopcr_overall_mean if args.datatype == 'original' else None,  # 전체 AOPCR 합 (original 데이터셋일 때만)
        M_expl.cpu().numpy(),           # class-wise explanation curves (C, n_steps)
        M_rand.cpu().numpy(),           # class-wise random curves (C, n_steps)
        alphas.cpu().numpy(),           # perturbation ratios (n_steps,)
        counts.cpu().numpy(),           # per-class bag counts (C,)
    )
