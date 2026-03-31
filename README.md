# AmbiguousMIL with TG-IPCL for Time Series Classification

Multiple Instance Learning (MIL) framework for **weakly-supervised temporal interpretation** of multivariate time series. The primary model is **AmbiguousMIL** — a pooling architecture with instance-prototype contrastive learning that learns both bag-level classification and instance-level interpretability from bag-level labels only.

## Setup

```bash
conda create -n mil python=3.9
conda activate mil
pip install -r requirements.txt
```

UEA benchmark datasets are automatically downloaded by `aeon` on first run and cached in `./data/`. If you already have the datasets elsewhere, use `--data_path /your/data/dir` to avoid re-downloading.

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

## Evaluate a Checkpoint

Use `eval_checkpoint.py` to evaluate a saved checkpoint without re-training. You can download pretrained checkpoints for UEA benchmark datasets [here](https://osf.io/u2zjf/overview?view_only=25d2092580fe4e95a8a767a827a7a690).

```bash
python eval_checkpoint.py \
  --checkpoint ./savemodel/InceptBackbone/PenDigits/exp_0/weights/best_AmbiguousMIL.pth \
  --dataset PenDigits \
  --model AmbiguousMIL \
  --datatype mixed
```

Add `--compute_aopcr` to also compute the AOPCR interpretability metric:

```bash
python eval_checkpoint.py \
  --checkpoint ./savemodel/InceptBackbone/PenDigits/exp_0/weights/best_AmbiguousMIL.pth \
  --dataset PenDigits \
  --model AmbiguousMIL \
  --datatype mixed \
  --compute_aopcr
```

If your datasets are stored in a custom location:

```bash
python eval_checkpoint.py \
  --checkpoint ./path/to/best_AmbiguousMIL.pth \
  --dataset PenDigits \
  --model AmbiguousMIL \
  --datatype mixed \
  --data_path /your/data/dir
```

## Outputs

Checkpoints are saved to `./savemodel/InceptBackbone/{DATASET}/exp_{N}/weights/best_{MODEL}.pth`.

After training, the script automatically:
1. Re-evaluates the best model on the test set
2. Computes class-wise **AOPCR** (Area Over Perturbation Curve relative to Random baseline)
