"""BirdCLEF 2026 - Multi-label bird audio classification.

Usage:
    python main.py explore           # Explore dataset
    python main.py train             # Train model
    python main.py predict           # Generate predictions
    python main.py validate          # Validate on train set
"""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="BirdCLEF 2026")
    parser.add_argument("command", choices=["explore", "train", "predict", "validate"],
                        help="Command to run")
    parser.add_argument("--args", nargs=argparse.REMAINDER, help="Extra args passed to command")
    args = parser.parse_args()

    cmd = args.command

    if cmd == "explore":
        from src.data.explore import main as explore_main
        explore_main()

    elif cmd == "train":
        sys.argv = ["train.py"] + (args.args or [])
        from src.train import main as train_main
        train_main()

    elif cmd == "predict":
        sys.argv = ["inference.py"] + (args.args or [])
        from src.inference import main as inference_main
        inference_main()

    elif cmd == "validate":
        sys.argv = ["inference.py", "--use_train_labels"] + (args.args or [])
        from src.inference import main as inference_main
        inference_main()


if __name__ == "__main__":
    main()
