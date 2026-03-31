# AmbiguousMIL with Contrastive Learning for Time Series Classification

Multiple Instance Learning (MIL) framework for **weakly-supervised temporal interpretation** of multivariate time series. The primary model is **AmbiguousMILwithCL** — a dual-pooling architecture with instance-prototype contrastive learning that learns both bag-level classification and time-point-level interpretability from bag-level labels only.

## Core Idea

Training data consists of **bags** (sequences) with only bag-level labels (no per-timestep annotation). The model learns which time segments belong to which class via:

1. **Attention pooling (TS branch)**: class-specific soft attention over time &rarr; bag prediction
2. **Conjunctive pooling (TP branch)**: instance classifier + attention weighting &rarr; per-timestep prediction
3. **Prototype contrastive loss**: instance embeddings are pulled toward learned class prototypes, pushed away from others

At test time the model produces both a bag classification and a time-point attribution map.

## Repository Structure

```
newMIL/
├── main_cl_exp.py          # Main training script (DDP + WandB)
├── run_train.sh            # Batch training script for all datasets
├── requirements.txt
│
├── models/
│   ├── expmil.py           # AmbiguousMILwithCL  (primary model)
│   ├── inceptiontime.py    # InceptionTime backbone
│   └── common.py           # Shared utilities
│
├── syntheticdataset.py     # Synthetic bag generation
├── mydataload.py           # AEON / NPZ dataset loader
├── dba_dataloader.py       # DBA driving dataset loader
│
├── compute_aopcr.py        # AOPCR interpretability metric
├── utils.py                # Directory / logging utilities
├── lookhead.py             # Lookahead optimizer wrapper
│
├── data/                   # Datasets (auto-downloaded by aeon)
└── savemodel/              # Trained model checkpoints
```

## Setup

```bash
conda create -n mil python=3.9
conda activate mil
pip install -r requirements.txt
```

## Quick Start

### Single GPU

```bash
python main_cl_exp.py \
  --dataset PenDigits \
  --model AmbiguousMIL \
  --datatype mixed \
  --num_epochs 1500 \
  --epoch_des 20 \
  --bag_loss_w 0.5 \
  --inst_loss_w 0.2 \
  --sparsity_loss_w 0.05 \
  --proto_loss_w 0.2
```

### Multi-GPU (DDP)

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29500 \
  main_cl_exp.py --dataset PenDigits --model AmbiguousMIL --datatype mixed
```

### Batch Run (all UCR datasets)

```bash
bash run_train.sh
```

## Outputs

Checkpoints are saved to `./savemodel/InceptBackbone/{DATASET}/exp_{N}/weights/best_{MODEL}.pth`.

After training, the script automatically:
1. Re-evaluates the best model on the test set
2. Computes class-wise **AOPCR** (Area Over Perturbation Curve relative to Random baseline)

All metrics are logged to [WandB](https://wandb.ai) under project `TimeMIL`.
