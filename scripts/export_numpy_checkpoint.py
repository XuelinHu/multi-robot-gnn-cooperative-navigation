#!/usr/bin/env python3
"""Export a trained torch checkpoint for ROS system-Python inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.input, map_location='cpu', weights_only=True)
    values = {key: value.detach().cpu().numpy() for key, value in checkpoint['model'].items()}
    values['method'] = np.asarray(checkpoint['method'])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **values)
    print(f'wrote {output} fields={len(values)}')


if __name__ == '__main__':
    main()

