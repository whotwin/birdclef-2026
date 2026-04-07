"""
CLI entry point for contrastive learning pipeline.

Usage:
    # Pretrain encoder with SimCLR
    python main.py pretrain

    # Linear evaluation (freeze backbone, train linear head)
    python main.py linear_eval --checkpoint runs_contrastive/.../best.pt

    # Full pipeline: pretrain + linear_eval
    python main.py all --checkpoint runs_contrastive/.../best.pt
"""
import argparse
import os
import yaml
import torch

from train import train as train_contrastive
from linear_eval import train_linear_eval


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Contrastive Learning for BirdCLEF")
    parser.add_argument("action", choices=["pretrain", "linear_eval", "all"],
                        help="pretrain: train SimCLR encoder | linear_eval: evaluate with frozen backbone | all: pretrain then linear_eval")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Pretrained checkpoint for linear_eval")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device (e.g., cpu, cuda)")
    parser.add_argument("--fold", type=int, default=0,
                        help="Fold index for linear_eval")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.device:
        cfg['training']['device'] = args.device

    if args.action == "pretrain":
        run_dir = train_contrastive(cfg)
        print(f"\nPretraining done! Checkpoints at: {run_dir}")

    elif args.action == "linear_eval":
        if not args.checkpoint:
            raise ValueError("--checkpoint required for linear_eval")
        auc = train_linear_eval(args.checkpoint, cfg, fold=args.fold)
        print(f"\nLinear Eval AUC: {auc:.4f}")

    elif args.action == "all":
        # Pretrain
        run_dir = train_contrastive(cfg)
        # Find best.pt
        ckpt_dir = os.path.join(run_dir, "checkpoints")
        ckpt_path = os.path.join(ckpt_dir, "best.pt")
        # Linear eval
        auc = train_linear_eval(ckpt_path, cfg, fold=args.fold)
        print(f"\nLinear Eval AUC: {auc:.4f}")


if __name__ == "__main__":
    main()
