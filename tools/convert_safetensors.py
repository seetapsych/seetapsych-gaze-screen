#!/usr/bin/env python3
"""Convert a PyTorch state dict (.pth/.pt) to safetensors format."""

from __future__ import annotations

import argparse
import os
import sys

import safetensors.torch
import torch


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert a PyTorch state dict file to safetensors format.")
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input PyTorch state dict file (.pth / .pt).",
    )
    parser.add_argument(
        "output_file",
        type=str,
        nargs="?",
        default=None,
        help="Path to the output safetensors file. If omitted, replaces input extension with .safetensors.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for map_location when loading the state dict. Defaults to 'cpu'.",
    )
    args = parser.parse_args()

    input_path: str = args.input_file
    if not os.path.isfile(input_path):
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        return 1

    if args.output_file is not None:
        output_path: str = args.output_file
    else:
        output_path = os.path.splitext(input_path)[0] + ".safetensors"

    print(f"Loading state dict from: {input_path}")
    state_dict = torch.load(input_path, map_location=args.device, weights_only=True)

    if not isinstance(state_dict, dict):
        print(f"Error: loaded object is not a dict, got {type(state_dict).__name__}", file=sys.stderr)
        return 1

    print(f"Saving safetensors to: {output_path}")
    safetensors.torch.save_file(state_dict, output_path)

    input_size = os.path.getsize(input_path)
    output_size = os.path.getsize(output_path)
    print(f"Done. Input: {input_size:,} bytes -> Output: {output_size:,} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
