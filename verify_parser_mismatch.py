#!/usr/bin/env python3
"""Compare gsplat's COLMAP parser against Difix3D's for pose parity.

Run this script after modifying either repo to confirm both parsers emit the
same camera poses, normalization transforms, intrinsics, and image ordering.

Example:
    python verify_parser_mismatch.py \
        --scene ../gs7/dataset/action-figure/iphone/train \
        --difix-root ../Difix3D
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


def _purge_dataset_modules() -> None:
    """Remove previously imported dataset modules so paths don't clash."""

    targets = []
    for name in list(sys.modules):
        if name.startswith("datasets") or name.startswith("examples.datasets"):
            targets.append(name)
        elif name.startswith("datasets.") or name.startswith("examples.datasets."):
            targets.append(name)
    for name in targets:
        sys.modules.pop(name, None)


def _import_parser(module_root: Path) -> type:
    module_root = module_root.resolve()
    if not module_root.exists():
        raise FileNotFoundError(f"parser path not found: {module_root}")
    _purge_dataset_modules()
    sys.path.insert(0, module_root.as_posix())
    try:
        module = importlib.import_module("datasets.colmap")
        Parser = getattr(module, "Parser")
    finally:
        sys.path.pop(0)
    return Parser


def _load_parser(
    label: str,
    module_root: Path,
    data_dir: Path,
    factor: int,
    normalize: bool,
    test_every: int,
) -> object:
    Parser = _import_parser(module_root)
    parser = Parser(
        data_dir=str(data_dir),
        factor=factor,
        normalize=normalize,
        test_every=test_every,
    )
    print(f"[{label}] loaded {len(parser.image_names)} images from {data_dir}")
    return parser


def _fetch_ordered_intrinsics(parser: object) -> List[np.ndarray]:
    arrays: List[np.ndarray] = []
    for cam_id in parser.camera_ids:
        arrays.append(np.asarray(parser.Ks_dict[cam_id], dtype=np.float32))
    return arrays


def _fetch_ordered_params(parser: object) -> List[np.ndarray]:
    params: List[np.ndarray] = []
    for cam_id in parser.camera_ids:
        params.append(np.asarray(parser.params_dict[cam_id], dtype=np.float32))
    return params


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return float("inf")
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a - b)))


def _compare_sequences(a: Iterable, b: Iterable) -> Tuple[bool, List[Tuple[int, object, object]]]:
    mismatches: List[Tuple[int, object, object]] = []
    for idx, (lhs, rhs) in enumerate(zip(a, b)):
        if lhs != rhs:
            mismatches.append((idx, lhs, rhs))
    length_mismatch = len(list(a)) != len(list(b))
    return (not mismatches and not length_mismatch), mismatches


def compare(gsplat_parser: object, difix_parser: object, atol: float) -> Dict[str, object]:
    report: Dict[str, object] = {}

    # Image ordering
    same_order = tuple(gsplat_parser.image_names) == tuple(difix_parser.image_names)
    report["image_order_equal"] = same_order
    if not same_order:
        report["image_order_diff"] = [(i, g, d) for i, (g, d) in enumerate(zip(gsplat_parser.image_names, difix_parser.image_names)) if g != d][:5]

    # Camera-to-world
    cam_diff = _max_abs_diff(
        np.asarray(gsplat_parser.camtoworlds, dtype=np.float32),
        np.asarray(difix_parser.camtoworlds, dtype=np.float32),
    )
    report["camtoworld_max_abs_diff"] = cam_diff
    report["camtoworld_match"] = cam_diff <= atol

    # Normalization transforms
    transform_diff = _max_abs_diff(
        np.asarray(gsplat_parser.transform, dtype=np.float32),
        np.asarray(difix_parser.transform, dtype=np.float32),
    )
    report["transform_max_abs_diff"] = transform_diff
    report["transform_match"] = transform_diff <= atol

    # Intrinsics
    g_intrinsics = np.stack(_fetch_ordered_intrinsics(gsplat_parser))
    d_intrinsics = np.stack(_fetch_ordered_intrinsics(difix_parser))
    k_diff = _max_abs_diff(g_intrinsics, d_intrinsics)
    report["intrinsics_max_abs_diff"] = k_diff
    report["intrinsics_match"] = k_diff <= atol

    # Distortion params
    g_params = np.stack(_fetch_ordered_params(gsplat_parser))
    d_params = np.stack(_fetch_ordered_params(difix_parser))
    param_diff = _max_abs_diff(g_params, d_params)
    report["distortion_max_abs_diff"] = param_diff
    report["distortion_match"] = param_diff <= atol

    # Image sizes
    g_sizes = np.array([gsplat_parser.imsize_dict[cid] for cid in gsplat_parser.camera_ids], dtype=np.int32)
    d_sizes = np.array([difix_parser.imsize_dict[cid] for cid in difix_parser.camera_ids], dtype=np.int32)
    report["size_match"] = np.array_equal(g_sizes, d_sizes)
    if not report["size_match"]:
        report["size_diff_examples"] = list(zip(g_sizes[:5], d_sizes[:5]))

    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", type=Path, required=True, help="Path to COLMAP dataset directory (e.g., ../gs7/dataset/scene/iphone/train)")
    ap.add_argument("--difix-root", type=Path, default=Path(__file__).resolve().parent.parent / "Difix3D", help="Path to Difix3D repo (defaults to ../Difix3D relative to this script)")
    ap.add_argument("--factor", type=int, default=1)
    ap.add_argument("--no-normalize", action="store_false", dest="normalize", help="Disable parser normalization (defaults to enabled)")
    ap.set_defaults(normalize=True)
    ap.add_argument("--test-every", type=int, default=8)
    ap.add_argument("--atol", type=float, default=1e-5, help="Absolute tolerance for numeric equality")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent
    gsplat_examples = repo_root / "examples"
    difix_examples = args.difix_root.resolve() / "examples" / "gsplat"

    print("=== Loading Parsers ===")
    gsplat_parser = _load_parser(
        label="gsplat",
        module_root=gsplat_examples,
        data_dir=args.scene.resolve(),
        factor=args.factor,
        normalize=args.normalize,
        test_every=args.test_every,
    )
    difix_parser = _load_parser(
        label="difix",
        module_root=difix_examples,
        data_dir=args.scene.resolve(),
        factor=args.factor,
        normalize=args.normalize,
        test_every=args.test_every,
    )

    print("\n=== Comparing Outputs ===")
    report = compare(gsplat_parser, difix_parser, atol=args.atol)
    for key, value in report.items():
        print(f"{key}: {value}")

    mismatches = [k for k, v in report.items() if k.endswith("match") and v is False]
    if mismatches:
        print("\n✗ Parsers differ (see metrics above).")
        sys.exit(1)
    print("\n✓ Parsers match within tolerance.")


if __name__ == "__main__":
    main()
