# dba_dataloader.py
# -*- coding: utf-8 -*-

import os
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset


# ============================================
# 1) DBA configuration
# ============================================

# Folder name -> class index
DBA_STYLE_TO_LABEL = {
    "aggressive": 0,
    "conservative": 1,
    "normal": 2,
}

# Feature columns to use from parsed_50hz.csv
# Modify to match actual csv column names.
DBA_FEATURE_COLS = [
    "imu_acc_long", "imu_acc_lat", "imu_yaw_rate","imu_roll_rate",
    "odom_vx", # vehicle speed
    "odom_wz", # vehicle angular velocity
    "scc_obj_relspd", "scc_obj_dst",
    # "lead_present"
]


# ============================================
# 2) Sequence scanning
# ============================================

def dba_scan_sequences(root_dir: str) -> List[Tuple[str, int]]:
    """
    Scans folders 1_xxx, 2_xxx, ... under root_dir and collects
    parsed_50hz.csv paths and labels from aggressive/conservative/normal subdirectories.

    return:
      [(csv_path, label_int), ...]
    """
    seq_list: List[Tuple[str, int]] = []

    for seq_name in sorted(os.listdir(root_dir)):
        seq_dir = os.path.join(root_dir, seq_name)
        if not os.path.isdir(seq_dir):
            continue

        for style_name, label_id in DBA_STYLE_TO_LABEL.items():
            style_dir = os.path.join(seq_dir, style_name)
            if not os.path.isdir(style_dir):
                continue

            csv_path = os.path.join(style_dir, "parsed_50hz.csv")
            if os.path.isfile(csv_path):
                seq_list.append((csv_path, label_id))

    return seq_list


# ============================================
# 3) Sliding window generation
# ============================================

def _build_windows_from_sequences(
    sequences: List[Tuple[str, int]],
    feature_cols: List[str],
    window_size: int,
    stride: int,
):
    """
    sequences: [(csv_path, label_int), ...]
    feature_cols: list of feature column names to use

    return:
      X: np.ndarray [N, window_size, D]
      y: np.ndarray [N] (int label)
    """
    X_list = []
    y_list = []

    for csv_path, label in sequences:
        df = pd.read_csv(csv_path)

        # Use only the feature subset
        X_seq = df[feature_cols].to_numpy(dtype=np.float32)  # [T, D]
        T = X_seq.shape[0]
        if T < window_size:
            continue

        for start in range(0, T - window_size + 1, stride):
            win = X_seq[start:start + window_size]  # [window_size, D]
            X_list.append(win)
            y_list.append(label)

    if len(X_list) == 0:
        raise RuntimeError(
            "No valid windows were generated for DBA dataset. "
            "Check window_size/stride/feature_cols."
        )

    X = np.stack(X_list, axis=0)  # [N, L, D]
    y = np.array(y_list, dtype=np.int64)
    return X, y

def _get_driver_id_from_csv(csv_path: str) -> str:
    """
    csv_path: .../<driver>/<style>/parsed_50hz.csv
    Uses the <driver> folder name as driver_id.
    """
    style_dir = os.path.dirname(csv_path)          # .../<driver>/<style>
    driver_dir = os.path.dirname(style_dir)        # .../<driver>
    driver_id = os.path.basename(driver_dir)       # e.g. "1_xxx"
    return driver_id


def _split_sequences_by_driver(
    seq_list: List[Tuple[str, int]],
    test_ratio: float,
    seed: int = 42,
):
    """
    seq_list: [(csv_path, label_int), ...]
    -> Groups by driver and performs train/test split.

    Assumption:
      - Each driver has one of each: aggressive/normal/conservative.
    Guarantee:
      - If there is at least one test driver, all 3 classes appear in test.
    """
    # driver_id -> [(csv_path, label_int), ...]
    driver_to_seqs = {}
    for csv_path, label in seq_list:
        driver_id = _get_driver_id_from_csv(csv_path)
        driver_to_seqs.setdefault(driver_id, []).append((csv_path, label))

    driver_ids = list(driver_to_seqs.keys())
    rng = random.Random(seed)
    rng.shuffle(driver_ids)

    n_driver_total = len(driver_ids)
    n_driver_test = max(1, int(round(n_driver_total * test_ratio)))
    n_driver_train = n_driver_total - n_driver_test
    if n_driver_train == 0:
        # extreme case: prevent all drivers from going to test
        n_driver_train = 1
        n_driver_test = n_driver_total - 1

    test_driver_ids  = set(driver_ids[:n_driver_test])
    train_driver_ids = set(driver_ids[n_driver_test:])

    train_seqs: List[Tuple[str, int]] = []
    test_seqs:  List[Tuple[str, int]] = []

    for d in train_driver_ids:
        train_seqs.extend(driver_to_seqs[d])
    for d in test_driver_ids:
        test_seqs.extend(driver_to_seqs[d])

    rng.shuffle(train_seqs)
    rng.shuffle(test_seqs)

    print(
        f"[DBA] drivers total: {n_driver_total}, "
        f"train: {len(train_driver_ids)}, test: {len(test_driver_ids)}"
    )

    train_short = sorted(d.split('_')[0] for d in train_driver_ids)
    test_short  = sorted(d.split('_')[0] for d in test_driver_ids)
    print(f"[DBA] train driver ids: {train_short}")
    print(f"[DBA] test  driver ids: {test_short}")

    return train_seqs, test_seqs

def _stratified_split_sequences(
    seq_list: List[Tuple[str, int]],
    test_ratio: float,
    seed: int = 42,
):
    """
    seq_list: [(csv_path, label_int), ...]
    Splits train/test proportionally per label,
    ensuring each label has at least one sample in the test set.
    """
    # label -> [(csv_path, label_int), ...]
    by_label = {}
    for path, label in seq_list:
        by_label.setdefault(label, []).append((path, label))

    rng = random.Random(seed)

    train_seqs = []
    test_seqs  = []

    for label, items in by_label.items():
        items = items[:]  # copy
        rng.shuffle(items)

        n_total = len(items)
        if n_total == 0:
            continue

        # Ensure each class has at least one sample in test
        n_test = max(1, int(round(n_total * test_ratio)))
        n_train = n_total - n_test

        # n_train may be 0 (when all samples of a class go to test)
        test_seqs.extend(items[:n_test])
        train_seqs.extend(items[n_test:])

    # Shuffle again overall for order randomization
    rng.shuffle(train_seqs)
    rng.shuffle(test_seqs)

    return train_seqs, test_seqs



# ============================================
# 4) Builder for TimeMIL
# ============================================

def build_dba_tensors(
    root_dir: str,
    feature_cols,
    window_size: int = 50,
    stride: int = 10,
    test_ratio: float = 0.2,
    seed: int = 42,
):
    """
    Splits DBA dataset into train/test at the sequence level,
    then slices each into sliding windows and returns as TensorDataset.

    return:
      trainset : TensorDataset( Xtr:[N_tr, L, D], ytr:[N_tr, C] )
      testset  : TensorDataset( Xte:[N_te, L, D], yte:[N_te, C] )
      seq_len      : L
      num_classes  : C (=3)
      feat_in      : D
    """
    seq_list = dba_scan_sequences(root_dir)
    if len(seq_list) == 0:
        raise RuntimeError(f"No parsed_50hz.csv found under: {root_dir}")

    # train_seqs, test_seqs = _stratified_split_sequences(
    #     seq_list, test_ratio=test_ratio, seed=seed
    # )
    train_seqs, test_seqs = _split_sequences_by_driver(
        seq_list, test_ratio=test_ratio, seed=seed
    )

    print(f"[DBA] total seqs: {len(seq_list)}, "
        f"train: {len(train_seqs)}, test: {len(test_seqs)}")

    Xtr, ytr = _build_windows_from_sequences(train_seqs, feature_cols, window_size, stride)
    Xte, yte = _build_windows_from_sequences(test_seqs, feature_cols, window_size, stride)

    num_classes = len(DBA_STYLE_TO_LABEL)
    seq_len = Xtr.shape[1]
    feat_in = Xtr.shape[2]

    Xtr_t = torch.from_numpy(Xtr)       # [N_tr, L, D]
    Xte_t = torch.from_numpy(Xte)
    ytr_idx = torch.from_numpy(ytr)     # [N_tr]
    yte_idx = torch.from_numpy(yte)

    ytr_oh = F.one_hot(ytr_idx, num_classes=num_classes).float()  # [N_tr, C]
    yte_oh = F.one_hot(yte_idx, num_classes=num_classes).float()

    trainset = TensorDataset(Xtr_t, ytr_oh)
    testset  = TensorDataset(Xte_t, yte_oh)

    return trainset, testset, seq_len, num_classes, feat_in


def build_dba_for_timemil(args):
    """
    Convenience wrapper for use in main_cl_fix.py.
    Uses args.dba_root, args.dba_window, args.dba_stride, args.dba_test_ratio, args.seed.
    """
    trainset, testset, seq_len, num_classes, feat_in = build_dba_tensors(
        root_dir=args.dba_root,
        feature_cols=DBA_FEATURE_COLS,
        window_size=args.dba_window,
        stride=args.dba_stride,
        test_ratio=args.dba_test_ratio,
        seed=args.seed,
    )
    return trainset, testset, seq_len, num_classes, feat_in

def _build_dba_base_sequences(
    root_dir: str,
    feature_cols: List[str],
    test_ratio: float = 0.2,
    seed: int = 42,
):
    """
    Builder for DBA 'base sequences'.
    - Treats each parsed_50hz.csv as a single sequence,
    - Zero-pads to the maximum length to create [N, L, D] tensors,
    - Returns train/test split at the sequence level.

    return:
      Xtr      : torch.FloatTensor [N_tr, L, D]
      ytr_idx  : torch.LongTensor  [N_tr]
      Xte      : torch.FloatTensor [N_te, L, D]
      yte_idx  : torch.LongTensor  [N_te]
      seq_len  : L
      num_cls  : C
      feat_in  : D
    """
    seq_list = dba_scan_sequences(root_dir)
    if len(seq_list) == 0:
        raise RuntimeError(f"No parsed_50hz.csv found under: {root_dir}")

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    len_list: List[int] = []

    for csv_path, label in seq_list:
        df = pd.read_csv(csv_path)
        X_seq = df[feature_cols].to_numpy(dtype=np.float32)  # [T, D]
        X_list.append(X_seq)
        y_list.append(label)
        len_list.append(X_seq.shape[0])

    max_len = max(len_list)
    D = X_list[0].shape[1]
    N = len(X_list)

    # zero-padding
    X_pad = np.zeros((N, max_len, D), dtype=np.float32)
    for i, X_seq in enumerate(X_list):
        L = X_seq.shape[0]
        X_pad[i, :L, :] = X_seq

    y_arr = np.array(y_list, dtype=np.int64)

    # Per-sequence train/test split
    indices = np.arange(N)
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)

    n_test = int(round(N * test_ratio))
    n_train = N - n_test
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    Xtr = torch.from_numpy(X_pad[train_idx])      # [N_tr, L, D]
    Xte = torch.from_numpy(X_pad[test_idx])       # [N_te, L, D]
    ytr_idx = torch.from_numpy(y_arr[train_idx])  # [N_tr]
    yte_idx = torch.from_numpy(y_arr[test_idx])   # [N_te]

    num_classes = len(DBA_STYLE_TO_LABEL)
    seq_len = max_len
    feat_in = D

    return Xtr, ytr_idx, Xte, yte_idx, seq_len, num_classes, feat_in


def build_dba_for_mixed(args):
    """
    Helper for --dataset dba, --datatype mixed in main_cl_fix.py.

    Returns the format required by MixedSyntheticBagsConcatK:
      - Xtr, Xte : [N, L, D]
      - ytr_idx, yte_idx : [N] (class index)
    """
    Xtr, ytr_idx, Xte, yte_idx, seq_len, num_classes, feat_in = _build_dba_base_sequences(
        root_dir=args.dba_root,
        feature_cols=DBA_FEATURE_COLS,
        test_ratio=args.dba_test_ratio,
        seed=args.seed,
    )
    return Xtr, ytr_idx, Xte, yte_idx, seq_len, num_classes, feat_in

def build_dba_windows_for_mixed(
    args,
):
    """
    Helper for DBA + datatype='mixed'.

    - Treats each parsed_50hz.csv as a sequence
    - Slices into window-level samples using window_size / stride
    - Separates train/test windows based on sequence-level split
    - Returns in a format suitable for MixedSyntheticBagsConcatK

    return:
      Xtr      : torch.FloatTensor [N_tr_win, L, D]
      ytr_idx  : torch.LongTensor  [N_tr_win]
      Xte      : torch.FloatTensor [N_te_win, L, D]
      yte_idx  : torch.LongTensor  [N_te_win]
      seq_len  : L (= window_size)
      num_cls  : C
      feat_in  : D
    """
    root_dir    = args.dba_root
    feature_cols = DBA_FEATURE_COLS
    window_size = args.dba_window
    stride      = args.dba_stride
    test_ratio  = args.dba_test_ratio
    seed        = args.seed

    seq_list = dba_scan_sequences(root_dir)
    if len(seq_list) == 0:
        raise RuntimeError(f"No parsed_50hz.csv found under: {root_dir}")

#     train_seqs, test_seqs = _stratified_split_sequences(
#         seq_list, test_ratio=test_ratio, seed=seed
# )
    train_seqs, test_seqs = _split_sequences_by_driver(
        seq_list, test_ratio=test_ratio, seed=seed
    )

    # Reuse existing _build_windows_from_sequences
    Xtr_np, ytr_np = _build_windows_from_sequences(
        train_seqs, feature_cols, window_size, stride
    )
    Xte_np, yte_np = _build_windows_from_sequences(
        test_seqs,  feature_cols, window_size, stride
    )

    num_classes = len(DBA_STYLE_TO_LABEL)
    seq_len = window_size
    feat_in = Xtr_np.shape[2]   # D

    Xtr = torch.from_numpy(Xtr_np)       # [N_tr_win, L, D]
    Xte = torch.from_numpy(Xte_np)       # [N_te_win, L, D]
    ytr_idx = torch.from_numpy(ytr_np)   # [N_tr_win]
    yte_idx = torch.from_numpy(yte_np)   # [N_te_win]

    return Xtr, ytr_idx, Xte, yte_idx, seq_len, num_classes, feat_in