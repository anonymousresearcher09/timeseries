import math
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.inceptiontime import InceptionTimeFeatureExtractor


class AmbiguousMILwithCL(nn.Module):
    """
    InceptionTime backbone + (Attention pooling for bag prediction)
                          + (Conjunctive pooling for time-point prediction)
                          + Contrastive projection between bag embedding & time-point embedding.
    """

    def __init__(
        self,
        in_features: int,
        n_classes: int,
        mDim: int = 128,
        dropout: float = 0.0,
        is_instance: bool = True,     # instance logits/attn을 함께 리턴할지 여부
    ) -> None:
        super().__init__()
        self.is_instance = is_instance
        self.n_classes = n_classes
        self.mDim = mDim

        # ------------------------------
        # 1) Backbone (MILLET 스타일)
        # ------------------------------
        out_channels = max(1, mDim // 4)
        self.feature_extractor = InceptionTimeFeatureExtractor(
            n_in_channels=in_features
        )

        # ------------------------------
        # 2) Attention pooling head (TS prediction)
        # ------------------------------
        self.attn_head_ts = nn.Sequential(
            nn.Linear(mDim, 8*n_classes),
            nn.Tanh(),
            nn.Linear(8*n_classes, n_classes),
            nn.Sigmoid(),
        )
        self.bag_classifier_ts = nn.Linear(mDim, 1)

        # ------------------------------
        # 3) Conjunctive pooling head (TP prediction)
        # ------------------------------
        self.attn_head_tp = nn.Sequential(
            nn.Linear(mDim, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )
        self.inst_classifier = nn.Linear(mDim, n_classes)

        self.dropout_p = dropout
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, warmup: bool = False):
        """
        Args:
            x: [B, T, D]
        Returns:
            bag_logits_ts:  [B, C]  (attention pooling 기반 TS prediction)
            bag_logits_tp:  [B, C]  (conjunctive pooling 기반 bag prediction)
            inst_logits:    [B, T, C] (time point prediction)
            attn_ts:        [B, T, 1]
            attn_tp:        [B, T, 1]
        """
        B, T, D_in = x.shape

        # 1) Backbone
        x = x.transpose(1, 2)             # [B, D_in, T]
        feats = self.feature_extractor(x) # [B, D, T]

        if self.dropout_p > 0:
            feats = self.dropout(feats)

        inst_emb = feats.transpose(1, 2)  # [B, T, mDim]

        # ------------------------------
        # 2) Attention pooling (TS)
        # ------------------------------
        score_ts = self.attn_head_ts(inst_emb)              # [B, T, C]
        score_ts = score_ts.permute(0, 2, 1)               # [B, C, T]
        attn_ts = F.softmax(score_ts, dim=-1)              # [B, C, T]

        bag_emb = torch.bmm(attn_ts, inst_emb)        # [B, C, mDim]
        B, C, D = bag_emb.shape
        bag_emb_flat = bag_emb.reshape(B * C, D)      # [B*C, mDim]
        bag_logits_flat = self.bag_classifier_ts(bag_emb_flat)  # [B*C, 1]
        bag_logits_ts = bag_logits_flat.view(B, C)    # [B, C]

        # ------------------------------
        # 3) Conjunctive pooling (TP)
        # ------------------------------
        attn_tp = self.attn_head_tp(inst_emb)           # [B, T, 1]
        inst_logits = self.inst_classifier(inst_emb)    # [B, T, C]
        weighted_logits = inst_logits * attn_tp         # [B, T, C]
        bag_logits_tp = weighted_logits.mean(dim=1)     # [B, C]

        if self.is_instance:
            return (
                bag_logits_ts,
                bag_logits_tp,
                weighted_logits,
                inst_logits,
                bag_emb,
                inst_emb,
                attn_ts,
                attn_tp,
            )
        else:
            return bag_logits_ts