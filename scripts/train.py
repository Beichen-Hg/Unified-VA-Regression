"""Train a baseline, foundation model, or conditional specialization model."""
import argparse
import _common
from unified_va.cli import add_run_arguments, main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_arguments(parser)
    args = parser.parse_args()
    command = ["train-" + args.stage]
    for key in ("split", "seed", "data_root", "model_dir", "run_dir", "device"):
        command.extend(["--" + key.replace("_", "-"), str(getattr(args, key))])
    if args.parent:
        command.extend(["--parent", args.parent])
    main(command)
