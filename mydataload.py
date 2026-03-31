# mydataload.py
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Optional, Sequence
from aeon.datasets import load_classification

# ------------------ Utilities ------------------
def _to_multi_hot_from_ts_int(y_ts_int: np.ndarray, C: int, exclude: Optional[Sequence[int]] = None) -> np.ndarray:
    """
    y_ts_int: (N, L) integer labels. Generates multi-hot from the union of classes present in each sequence.
    exclude : list of class IDs to exclude (e.g., [0])
    return  : (N, C) float32 multi-hot
    """
    N, L = y_ts_int.shape
    y_seq = np.zeros((N, C), dtype=np.float32)
    if exclude is None:
        exclude = []
    excl = set(int(x) for x in exclude)

    for i in range(N):
        uniq = np.unique(y_ts_int[i])
        uniq = [int(u) for u in uniq if int(u) not in excl and 0 <= int(u) < C]
        if uniq:
            y_seq[i, uniq] = 1.0
    return y_seq

def _to_multi_hot_from_ts_onehot(y_ts_oh: np.ndarray, exclude: Optional[Sequence[int]] = None) -> np.ndarray:
    """
    y_ts_oh: (N, L, C) one-hot timestep labels -> (N, C) multi-hot
    exclude: classes to exclude
    """
    N, L, C = y_ts_oh.shape
    y_seq = (y_ts_oh.sum(axis=1) > 0).astype(np.float32)  # (N, C)
    if exclude:
        y_seq[:, list(exclude)] = 0.0
    return y_seq

# --------------- NPZ Window Dataset ---------------
class NPZWindowDataset(Dataset):
    """
    X: (N, L, D)
    y_seq: (N, C)  <- multi-hot (sequence-level) returned for training
    (optional) y_ts_int: (N, L)  timestep integer labels
    (optional) y_ts_oh : (N, L, C) timestep one-hot labels
    """
    def __init__(self, X: np.ndarray,
                 y_seq: Optional[np.ndarray],
                 y_ts_int: Optional[np.ndarray],
                 y_ts_oh: Optional[np.ndarray],
                 exclude_labels: Optional[Sequence[int]] = None):
        super().__init__()
        assert X.ndim == 3, f"X shape must be (N,L,D), got {X.shape}"
        self.X = X.astype(np.float32, copy=False)
        self.N, self.L, self.D = self.X.shape

        self.y_ts_int = y_ts_int  # (N, L) or None
        self.y_ts_oh  = y_ts_oh   # (N, L, C) or None
        self.exclude  = list(exclude_labels) if exclude_labels else []

        if y_seq is not None:
            # Use existing sequence-level labels as-is
            self.y_seq = y_seq.astype(np.float32, copy=False)
        else:
            # Generate sequence multi-hot from timestep labels
            if self.y_ts_oh is not None:
                C = self.y_ts_oh.shape[-1]
                self.y_seq = _to_multi_hot_from_ts_onehot(self.y_ts_oh, exclude=self.exclude)
            elif self.y_ts_int is not None:
                # Estimate C as max_label+1 (safer to use len of 'label_values' from npz if available)
                C = int(self.y_ts_int.max()) + 1 if self.y_ts_int.size else 0
                self.y_seq = _to_multi_hot_from_ts_int(self.y_ts_int, C=C, exclude=self.exclude)
            else:
                raise ValueError("Neither sequence labels nor timestep labels were found in NPZ.")

        assert self.y_seq.ndim == 2 and self.y_seq.shape[0] == self.N, \
            f"y_seq must be (N,C), got {self.y_seq.shape}"
        self.C = self.y_seq.shape[1]

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])           # (L, D)
        y = torch.from_numpy(self.y_seq[idx])       # (C,)  <- training uses only weak supervision (multi-hot)
        return x, y

    # Optional helper for retrieving timestep labels during validation
    def get_timestep_labels(self, idx):
        if self.y_ts_oh is not None:
            return self.y_ts_oh[idx]    # (L, C)
        if self.y_ts_int is not None:
            return self.y_ts_int[idx]   # (L,)
        return None

# --------------- AEON + NPZ branching loadorean ---------------
class loadorean(Dataset):
    """
    AEON datasets are used as-is.
    When NPZ is provided:
      - Priority: use y_seq_* directly if available
      - Otherwise, if y_ts_* exists, generate multi-hot from the union of labels in the window (in the loader)
    To exclude labels, pass args.exclude_labels = "0,255" format (ignored if absent).
    """
    def __init__(self, args, split='train', seed=0, return_instance_labels=False):
        super().__init__()
        self.args = args
        self.split = split
        self.return_instance_labels = return_instance_labels

        # ========== NPZ branch ==========
        if args.dataset in ['PAMAP2']:
            data = np.load(args.prepared_npz, allow_pickle=False)

            # Per-split key mapping
            key_map = {
                "train": ("X_train", "y_train", "y_ts_train", "y_ts_train_oh"),
                "val":   ("X_val",   "y_val",   "y_ts_val",   "y_ts_val_oh"),
                "valid": ("X_val",   "y_val",   "y_ts_val",   "y_ts_val_oh"),
                "test":  ("X_test",  "y_test",  "y_ts_test",  "y_ts_test_oh"),
            }
            k = "val" if split in ("val", "valid", "validation") else ("test" if split=="test" else "train")
            X_key, Y_key, YTS_key, YTSOH_key = key_map[k]

            X = data[X_key]                                  # (N, L, D)
            y_seq = data[Y_key] if Y_key in data.files else None
            y_ts_int = data[YTS_key] if YTS_key in data.files else None
            y_ts_oh  = data[YTSOH_key] if YTSOH_key in data.files else None

            self._inner = NPZWindowDataset(
                X=X, y_seq=y_seq, y_ts_int=y_ts_int, y_ts_oh=y_ts_oh
            )

            # External attributes (original code compatibility)
            self.max_len   = int(data["seq_len"]) if "seq_len" in data.files else self._inner.L
            self.num_class = self._inner.C
            self.feat_in   = self._inner.D
            self._mode = "npz"
            return

        # ========== AEON branch (existing) ==========
        elif args.dataset == 'JapaneseVowels':
            self.seq_len = 29
        elif args.dataset == 'SpokenArabicDigits':
            self.seq_len = 93
        elif args.dataset == 'CharacterTrajectories':
            self.seq_len = 182
        elif args.dataset == 'InsectWingbeat':
            self.seq_len = 78

        if split == 'train':
            if args.dataset == 'InsectWingbeat':
                Xtr, ytr, meta = load_classification(name='InsectWingbeat', split='train', extract_path='../timeclass/dataset/')
            else:
                Xtr, ytr, meta = load_classification(name=args.dataset, split='train')
            word_to_idx = {meta['class_values'][i]: i for i in range(len(meta['class_values']))}
            ytr = [word_to_idx[i] for i in ytr]
            self.label = F.one_hot(torch.tensor(ytr)).float()
            self.FeatList = Xtr
        elif split == 'test':
            if args.dataset == 'InsectWingbeat':
                Xte, yte, meta = load_classification(name='InsectWingbeat', split='test', extract_path='../timeclass/dataset/')
            else:
                Xte, yte, meta = load_classification(name=args.dataset, split='test')
            word_to_idx = {meta['class_values'][i]: i for i in range(len(meta['class_values']))}
            yte = [word_to_idx[i] for i in yte]
            self.label = F.one_hot(torch.tensor(yte)).float()
            self.FeatList = Xte
        else:
            raise ValueError(f"Unknown split: {split}")

        self.feat_in  = self.FeatList[0].shape[0]
        self.max_len  = self.seq_len
        self.num_class= self.label.shape[-1]
        self._mode = "aeon"

    def __len__(self):
        if getattr(self, "_mode", "") == "npz":
            return len(self._inner)
        else:
            return len(self.label)
        
    def __getitem__(self, idx):
        if getattr(self, "_mode", "") == "npz":
            x = torch.from_numpy(self._inner.X[idx])            # (L, D)
            y = torch.from_numpy(self._inner.y_seq[idx])        # (C,)

            if self.return_instance_labels:
                # 1) One-hot timestep labels already available (N, L, C)
                if self._inner.y_ts_oh is not None:
                    inst = torch.from_numpy(self._inner.y_ts_oh[idx]).float()   # (L, C)

                # 2) Only integer timestep labels available (N, L)
                elif self._inner.y_ts_int is not None:
                    ints = torch.from_numpy(self._inner.y_ts_int[idx]).long()   # (L,)
                    inst = F.one_hot(ints, num_classes=self._inner.C).float()    # (L, C)

                else:
                    inst = None

                return x, y, inst  # inst: (L, C)

            return x, y

        # AEON branch (no instance labels)
        feats = torch.from_numpy(self.FeatList[idx]).permute(1, 0).float()  # (L, D)
        if feats.shape[0] < self.max_len:
            feats = F.pad(feats, pad=(0, 0, self.max_len - feats.shape[0], 0))
        label = self.label[idx].float()
        return feats, label

    # Optional: use this method to retrieve timestep labels during validation
    def get_timestep_labels(self, idx):
        if getattr(self, "_mode", "") == "npz":
            return self._inner.get_timestep_labels(idx)
        return None

    def proterty(self):
        return self.max_len, self.num_class, self.feat_in
