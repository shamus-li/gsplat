#!/usr/bin/env python3
"""Emit train/test alignment metadata for downstream evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional, Tuple

import numpy as np


def _make_parser(path: Path, test_every: int):
    examples_dir = Path(__file__).resolve().parent
    if examples_dir.as_posix() not in sys.path:
        sys.path.insert(0, examples_dir.as_posix())

    from datasets.colmap import Parser  # type: ignore

    return Parser(
        data_dir=str(path),
        factor=1,
        normalize=True,
        test_every=test_every,
    )


def compute_alignment(
    train_dir: Path,
    subset_dir: Optional[Path],
    train_test_every: int,
    eval_test_every: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (align, train_norm, subset_norm).

    align_transform maps cameras normalized with ``subset_norm`` into the
    training coordinate frame defined by ``train_norm``. When ``subset_dir`` is
    omitted we fall back to legacy behavior and only persist the train
    normalization (align becomes identity and support == base).
    """

    train_parser = _make_parser(train_dir, train_test_every)
    base = train_parser.transform.astype(np.float32)

    if subset_dir is None:
        return np.eye(4, dtype=np.float32), base, base.copy()

    subset_parser = _make_parser(subset_dir, eval_test_every)
    support = subset_parser.transform.astype(np.float32)
    align = base @ np.linalg.inv(support)
    return align.astype(np.float32), base, support


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument(
        "--subset-dir",
        type=Path,
        default=None,
        help="Optional eval subset directory; required to compute non-identity alignments.",
    )
    parser.add_argument("--train-test-every", type=int, required=True)
    parser.add_argument(
        "--eval-test-every",
        type=int,
        default=1,
        help="test_every used for the eval subset (defaults to 1).",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    align, base_transform, support_transform = compute_alignment(
        train_dir=args.train_dir.resolve(),
        subset_dir=args.subset_dir.resolve() if args.subset_dir else None,
        train_test_every=args.train_test_every,
        eval_test_every=args.eval_test_every,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        align_transform=align.astype(np.float32),
        base_transform=base_transform.astype(np.float32),
        support_transform=support_transform.astype(np.float32),
    )
    print(f"Wrote alignment metadata to {args.output.resolve()}")


if __name__ == "__main__":
    main()
