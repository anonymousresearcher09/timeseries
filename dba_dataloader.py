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
# 1) DBA용 설정
# ============================================

# 폴더 이름 -> class index
DBA_STYLE_TO_LABEL = {
    "aggressive": 0,
    "conservative": 1,
    "normal": 2,
}

# parsed_50hz.csv 내부에서 사용할 feature 컬럼들
# 실제 csv 컬럼 이름에 맞게 수정해서 쓰세요.
DBA_FEATURE_COLS = [
    "imu_acc_long", "imu_acc_lat", "imu_yaw_rate","imu_roll_rate",
    "odom_vx", # 차량 속도
    "odom_wz", # 차량 각속도
    "scc_obj_relspd", "scc_obj_dst",
    # "lead_present"
]


# ============================================
# 2) 시퀀스 스캔
# ============================================

def dba_scan_sequences(root_dir: str) -> List[Tuple[str, int]]:
    """
    root_dir 아래 1_xxx, 2_xxx, ... 폴더에서
    aggressive/conservative/normal 하위의 parsed_50hz.csv 경로와 label을 수집.

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
# 3) sliding window 생성
# ============================================

def _build_windows_from_sequences(
    sequences: List[Tuple[str, int]],
    feature_cols: List[str],
    window_size: int,
    stride: int,
):
    """
    sequences: [(csv_path, label_int), ...]
    feature_cols: 사용할 feature 컬럼 이름 리스트

    return:
      X: np.ndarray [N, window_size, D]
      y: np.ndarray [N] (int label)
    """
    X_list = []
    y_list = []

    for csv_path, label in sequences:
        df = pd.read_csv(csv_path)

        # feature subset만 사용
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
    여기서 <driver> 폴더 이름을 driver_id로 사용.
    """
    style_dir = os.path.dirname(csv_path)          # .../<driver>/<style>
    driver_dir = os.path.dirname(style_dir)        # .../<driver>
    driver_id = os.path.basename(driver_dir)       # "1_xxx" 같은 이름
    return driver_id


def _split_sequences_by_driver(
    seq_list: List[Tuple[str, int]],
    test_ratio: float,
    seed: int = 42,
):
    """
    seq_list: [(csv_path, label_int), ...]
    -> driver 단위로 묶어서 train/test split.

    전제:
      - 각 driver가 aggressive/normal/conservative 1개씩 갖고 있음.
    보장:
      - test drivers가 1명 이상이면 test에 3개 class 모두 등장.
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
        # extreme case: 모든 driver가 test로 가는 경우 방지
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
    각 label 별로 비율에 맞춰 train/test를 나누되,
    각 label이 test에 최소 1개씩은 들어가도록 하는 split.
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

        # 각 클래스가 test에 최소 1개는 들어가도록 보장
        n_test = max(1, int(round(n_total * test_ratio)))
        n_train = n_total - n_test

        # n_train이 0이 될 수도 있음 (해당 클래스 전체가 test로 가는 경우)
        test_seqs.extend(items[:n_test])
        train_seqs.extend(items[n_test:])

    # 전체적으로도 다시 한 번 섞어주는 것이 좋습니다 (순서 랜덤화)
    rng.shuffle(train_seqs)
    rng.shuffle(test_seqs)

    return train_seqs, test_seqs



# ============================================
# 4) TimeMIL용 빌더
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
    DBA 데이터셋을 시퀀스 단위로 train/test split 하고,
    각각을 sliding window로 자른 뒤 TensorDataset으로 반환.

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
    main_cl_fix.py에서 편하게 쓰기 위한 wrapper.
    args.dba_root, args.dba_window, args.dba_stride, args.dba_test_ratio, args.seed 사용.
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
    DBA용 'base 시퀀스' 빌더.
    - 각 parsed_50hz.csv 전체를 하나의 시퀀스로 보고,
    - 최대 길이에 맞춰 zero-padding 해서 [N, L, D] 텐서를 만든 뒤
    - train/test를 sequence 단위로 나눠서 반환.

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

    # sequence 단위 train/test split
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
    main_cl_fix.py에서 --dataset dba, --datatype mixed 일 때 사용할 helper.

    MixedSyntheticBagsConcatK에서 요구하는 형태:
      - Xtr, Xte : [N, L, D]
      - ytr_idx, yte_idx : [N] (클래스 index)

    을 반환한다.
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
    DBA + datatype='mixed' 용 helper.

    - 각 parsed_50hz.csv 를 sequence로 보고
    - window_size / stride 로 잘라서 window 단위의 sample을 만든 뒤
    - train/test seq split 기준으로 train 윈도우 / test 윈도우 분리
    - MixedSyntheticBagsConcatK 에 넣기 좋은 형태로 반환

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

    # 기존 _build_windows_from_sequences 재사용
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