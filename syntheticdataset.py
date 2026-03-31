import torch
from torch.utils.data import Dataset
import numpy as np
import random
from typing import Dict, List, Tuple, Union

seed = 42

random.seed(seed)             # python random
np.random.seed(seed)          # numpy random
torch.manual_seed(seed)       # CPU
torch.cuda.manual_seed(seed)  # GPU 단일
torch.cuda.manual_seed_all(seed)  # multi-GPU

# 재현성을 위한 옵션들
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def _rand_partition(total_len: int, k: int, min_seg: int = 16) -> List[int]:
    assert k * min_seg <= total_len
    parts = np.random.dirichlet(np.ones(k))
    parts = (parts * (total_len - k * min_seg)).astype(int)
    rem = (total_len - k * min_seg) - parts.sum()
    parts[:rem] += 1
    parts = (parts + min_seg).tolist()
    parts[-1] += (total_len - sum(parts))
    return parts

class MixedSyntheticBags(Dataset):
    def __init__(self, X, y_idx, num_classes, total_bags=5000,
                 probs=None, total_len=None, min_seg_len=32,
                 ensure_distinct_classes=True, seed=42,
                 return_instance_labels=False):
        super().__init__()
        assert X.dim() == 3
        self.X = X
        self.y_idx = y_idx
        self.num_classes = num_classes
        self.total_bags = int(total_bags)
        self.min_seg_len = int(min_seg_len)
        self.ensure_distinct_classes = ensure_distinct_classes
        self.return_instance_labels = return_instance_labels

        if probs is None:
            probs = {'orig': 0.4, 2: 0.4, 3: 0.2}
        keys = list(probs.keys())
        vals = np.array([probs[k] for k in keys], dtype=float)
        vals = vals / vals.sum()
        self.modes = keys
        self.mode_probs = vals

        # 시드 고정
        random.seed(seed); np.random.seed(seed)
        self.N, self.T, self.D = X.shape
        self.total_len = self.T if total_len is None else int(total_len)

        # 클래스별 인덱스
        self.cls2inds = {c: [] for c in range(num_classes)}
        for i, c in enumerate(y_idx.tolist()):
            self.cls2inds[c].append(i)
        for c in range(num_classes):
            if len(self.cls2inds[c]) == 0:
                raise ValueError(f"class {c} has no samples.")

        # ---- 모든 bag의 스펙을 미리 샘플링(재현성 보장) ----
        self.sample_modes = np.random.choice(self.modes, size=self.total_bags, p=self.mode_probs)
        self.bag_specs = []  # 각 bag: dict 저장
        for mode in self.sample_modes:
            if mode == 'orig':
                k = 1
                classes = [random.randrange(self.num_classes)]
            else:
                k = int(mode)
                if self.ensure_distinct_classes and k <= self.num_classes:
                    classes = random.sample(range(self.num_classes), k)
                else:
                    classes = [random.randrange(self.num_classes) for _ in range(k)]
            parts = _rand_partition(self.total_len, k, min_seg=self.min_seg_len)

            seg_specs = []
            for c, L in zip(classes, parts):
                src_idx = random.choice(self.cls2inds[c])
                T_src = self.X[src_idx].shape[0]
                if L <= T_src:
                    s = random.randint(0, T_src - L)
                    tiled = False
                else:
                    s = -1               # 타일링 표시
                    tiled = True
                seg_specs.append({"cls": c, "seg_len": L, "src_idx": src_idx, "start": s, "tiled": tiled})
            self.bag_specs.append({"mode": mode, "classes": classes, "parts": parts, "segs": seg_specs})

    def __len__(self):
        return self.total_bags

    def __getitem__(self, idx):
        spec = self.bag_specs[idx]
        segs = []
        y_bag = torch.zeros(self.num_classes, dtype=torch.float32)
        # 타임스텝 단 클래스 id (후에 one-hot로 확장)
        class_per_step = torch.empty(self.total_len, dtype=torch.long)

        offset = 0
        meta_segments = []  # interpretability용 경계/원본 정보
        for seg in spec["segs"]:
            c, L, src_idx, s, tiled = seg["cls"], seg["seg_len"], seg["src_idx"], seg["start"], seg["tiled"]
            x_src = self.X[src_idx]  # [T_src, D]
            if tiled:
                rep = int(np.ceil(L / x_src.shape[0]))
                x_piece = x_src.repeat((rep, 1))[:L]
            else:
                x_piece = x_src[s:s+L]
            segs.append(x_piece)
            y_bag[c] = 1.0

            # 타임스텝 라벨 기록
            class_per_step[offset:offset+L] = c

            # 메타 기록 (bag 상의 위치, 원본에서의 위치/타일링 여부)
            meta_segments.append({
                "class": int(c),
                "bag_start": int(offset),
                "bag_end": int(offset + L),  # end-exclusive
                "src_idx": int(src_idx),
                "src_start": int(s),         # -1이면 tiled
                "tiled": bool(tiled)
            })
            offset += L

        bag = torch.cat(segs, dim=0)  # [total_len, D]

        if self.return_instance_labels:
            # [T, C] one-hot
            y_inst = torch.zeros(self.total_len, self.num_classes, dtype=torch.float32)
            y_inst[torch.arange(self.total_len), class_per_step] = 1.0
            # (bag, y_bag, y_inst, meta) 반환
            return bag, y_bag, y_inst
        else:
            return bag, y_bag

class MixedSyntheticBagsConcatK(Dataset):
    """
    X: [N, T, D]   - 원본 시퀀스들
    y_idx: [N]     - 각 시퀀스의 클래스 인덱스 (0 ~ num_classes-1)

    synthetic bag 하나는:
      - 원본 시퀀스 concat_k개를 선택해서 시간축으로 그대로 이어붙임
      - 길이 = concat_k * T
      - bag-level label은 안에 등장한 클래스들의 multi-hot
      - instance-level label은 timestep마다 해당 시퀀스의 클래스 one-hot

    특징:
      - 어떤 원본 시퀀스도 "자르지" 않습니다.
      - 클래스는 랜덤으로 뽑히므로
        -> 한 bag 안에 서로 다른 클래스 개수는 1 ~ concat_k 개 사이에서 자연스럽게 결정됩니다.
    """

    def __init__(
        self,
        X: torch.Tensor,
        y_idx: torch.Tensor,
        num_classes: int,
        total_bags: int = 5000,
        concat_k: int = 3,
        seed: int = 42,
        return_instance_labels: bool = False,
        with_replacement: bool = True,
    ):
        super().__init__()
        assert X.dim() == 3, "X는 [N, T, D] 텐서여야 합니다."
        self.X = X
        self.y_idx = y_idx
        self.num_classes = int(num_classes)
        self.total_bags = int(total_bags)
        self.concat_k = int(concat_k)
        self.return_instance_labels = return_instance_labels
        self.with_replacement = with_replacement

        self.N, self.T, self.D = X.shape
        self.total_len = self.concat_k * self.T  # bag 길이

        # 시드 고정 (dataset 내부 재현성)
        random.seed(seed)
        np.random.seed(seed)

        # ---- 미리 어떤 원본 시퀀스들을 이어붙일지 전체 bag에 대해 샘플링 ----
        # bag_specs[b] = [src_idx_1, src_idx_2, ..., src_idx_K]
        self.bag_specs: List[List[int]] = []

        indices = list(range(self.N))

        for _ in range(self.total_bags):
            if self.with_replacement:
                # 중복 허용 샘플링
                chosen = [random.choice(indices) for _ in range(self.concat_k)]
            else:
                # 중복 비허용 (N >= concat_k 가정)
                chosen = random.sample(indices, self.concat_k)

            self.bag_specs.append(chosen)

    def __len__(self):
        return self.total_bags

    def __getitem__(self, idx):
        src_indices = self.bag_specs[idx]  # 길이 concat_k

        bag_segments = []
        y_bag = torch.zeros(self.num_classes, dtype=torch.float32)

        # timestep별 클래스 id (나중에 one-hot으로 바꿀 수 있음)
        class_per_step = torch.empty(self.total_len, dtype=torch.long)

        offset = 0
        for src_idx in src_indices:
            src_idx = int(src_idx)
            x_src = self.X[src_idx]            # [T, D]
            c = int(self.y_idx[src_idx])      # 스칼라 클래스

            bag_segments.append(x_src)
            y_bag[c] = 1.0  # bag-level multi-hot

            # 이 시퀀스 구간의 timestep별 클래스
            class_per_step[offset:offset + self.T] = c
            offset += self.T

        # [concat_k * T, D]
        bag = torch.cat(bag_segments, dim=0)
        assert bag.shape[0] == self.total_len

        if self.return_instance_labels:
            # [T_total, C] one-hot
            y_inst = torch.zeros(self.total_len, self.num_classes, dtype=torch.float32)
            y_inst[torch.arange(self.total_len), class_per_step] = 1.0
            return bag, y_bag, y_inst
        else:
            return bag, y_bag