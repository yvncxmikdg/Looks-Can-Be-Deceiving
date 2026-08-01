#!/usr/bin/env python3
"""Install this repo's dependencies in the order pip -r requirements.txt can't.

Why this exists (not just `pip install -r requirements.txt`):
  1. setuptools must be capped (<70) *before* anything builds. mmcv 2.1.0's
     setup.py does `import pkg_resources`, which setuptools>=81 removed. A line
     in requirements.txt can't guarantee it lands first, and build isolation
     would ignore the env's setuptools anyway.
  2. torch must be installed *before* mmcv so mmcv matches/links against it.
  3. mmcv is pulled from a prebuilt wheel index keyed to the torch minor
     version (torch2.1), with --no-build-isolation so no source build happens.
  4. Only then does `-r requirements.txt` run; mmcv is already satisfied.

This is the single source of truth. setup_env.sh / setup_env.bat are thin
wrappers that just exec this file, so they can't drift from the real logic.

Usage:
    python setup_env.py [cpu|cu121]     # default: cu121
    ./setup_env.sh cpu
    setup_env.bat cu121
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REQUIREMENTS = os.path.join(HERE, "requirements.txt")

VARIANTS = {
    "cpu": {
        "torch_index": "https://download.pytorch.org/whl/cpu",
        "mmcv_index": "https://download.openmmlab.com/mmcv/dist/cpu/torch2.1/index.html",
        "torch_pkgs": ["torch", "torchvision", "torchaudio"],
    },
    "cu121": {
        "torch_index": "https://download.pytorch.org/whl/cu121",
        "mmcv_index": "https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html",
        "torch_pkgs": ["torch==2.1.2", "torchvision==0.16.2"],
    },
}


def run(args):
    """Print and execute a command, raising on non-zero exit."""
    print(f"+ {' '.join(args)}", flush=True)
    subprocess.check_call(args)


def main():
    """Parse the variant argument and run the ordered install sequence."""
    variant = sys.argv[1] if len(sys.argv) > 1 else "cu121"
    if variant not in VARIANTS:
        sys.exit(f"unknown variant {variant!r} (use one of: {', '.join(VARIANTS)})")
    cfg = VARIANTS[variant]
    pip = [sys.executable, "-m", "pip"]

    print(f"Setting up environment for variant: {variant}", flush=True)
    run(pip + ["install", "--upgrade", "pip"])
    run(pip + ["install", "setuptools<70", "wheel==0.46.3"])
    run(pip + ["install", *cfg["torch_pkgs"], "--index-url", cfg["torch_index"]])
    run(
        pip
        + [
            "install",
            "mmcv==2.1.0",
            "-f",
            cfg["mmcv_index"],
            "--no-build-isolation",
        ]
    )
    run(pip + ["install", "-r", REQUIREMENTS, "--no-build-isolation"])
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
