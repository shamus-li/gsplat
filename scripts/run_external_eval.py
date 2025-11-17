#!/usr/bin/env python3
"""
Run a short eval-only pass on an external COLMAP dataset using an existing
training checkpoint, regenerating PNG renders (unmasked) while keeping
metrics behavior unchanged.

Usage examples
--------------

1) Minimal: find latest ckpt in a training result dir and evaluate on EVAL_DIR
   with 10 closest eval frames, PNGs only (no video):

    python scripts/run_external_eval.py \
        --train_result_dir results/static/my_scene/iphone \
        --eval_dir /path/to/scene/static/shared/subsets/iphone_eval

2) Use all eval frames and keep covisible-masked metrics (renders remain
   unmasked due to trainer change) and provide an explicit covisible cache:

    python scripts/run_external_eval.py \
        --train_result_dir results/static/my_scene/iphone \
        --eval_dir /path/to/scene/static/shared/subsets/iphone_eval \
        --k 0 \
        --use_covisible --covisible_dir /path/to/scene/static/shared/covisible/iphone/1x

Notes
-----
- This script does not retrain. It loads the latest training checkpoint found
  under `<train_result_dir>/ckpts/` and runs `examples/simple_trainer.py` in
  eval-only mode on the external dataset.
- Saved PNGs show full renders (no covisible masking). If `--use_covisible` is
  provided, metrics remain masked by covisible regions.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

CKPT_PATTERN = re.compile(r"ckpt_(\d+)_rank\d+\.pt$")


def _latest_ckpt(result_dir: Path) -> Path:
    ckpt_dir = result_dir / "ckpts"
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
    ckpts = list(ckpt_dir.glob("ckpt_*_rank0.pt"))
    if not ckpts:
        raise FileNotFoundError(
            f"No checkpoints matching 'ckpt_*_rank0.pt' in {ckpt_dir}"
        )
    best = None
    best_step = -1
    for path in ckpts:
        match = CKPT_PATTERN.match(path.name)
        step = int(match.group(1)) if match else -1
        if step > best_step:
            best_step = step
            best = path
    assert best is not None
    return best


def _pose_opt_enabled(result_dir: Path) -> bool:
    cfg_path = result_dir / "cfg.yml"
    if not cfg_path.is_file():
        return False
    try:
        with cfg_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip().lower()
                if not stripped.startswith("pose_opt:"):
                    continue
                token = stripped.split(":", 1)[1].strip().strip("'\"")
                return token in {"true", "1", "yes"}
    except OSError:
        return False
    return False


def _covisible_alignment_candidates(data_dir: Path) -> List[Path]:
    candidates: List[Path] = []
    for ancestor in [data_dir, *data_dir.parents]:
        covi_root = ancestor / "covisible"
        if covi_root not in candidates:
            candidates.append(covi_root)
        subset_dir = covi_root / data_dir.name
        if subset_dir not in candidates:
            candidates.append(subset_dir)
    return candidates


def _dataset_alignment(train_dir: Path, eval_dir: Path) -> Optional[Path]:
    for base in (eval_dir, train_dir):
        for covi_dir in _covisible_alignment_candidates(base):
            candidate = covi_dir / "alignment.npz"
            if candidate.is_file():
                return candidate
    return None


def _shared_alignment_candidate(
    train_dir: Optional[Path], eval_dir: Path
) -> Optional[Path]:
    """Look for static-prep alignments (shared/alignments/<camera>_to_eval.npz, etc.)."""
    if train_dir is None:
        return None
    try:
        subsets_dir = train_dir.parent
        if subsets_dir.name != "subsets":
            return None
        prep_root = subsets_dir.parent
        align_root = prep_root / "alignments"
        if not align_root.is_dir():
            return None
        eval_name = eval_dir.name
        candidates = [
            align_root / f"{train_dir.name}_to_{eval_name}.npz",
        ]
        if eval_name.endswith("_eval"):
            candidates.append(align_root / f"{train_dir.name}_to_eval.npz")
        candidates.append(align_root / f"{eval_name}_to_{train_dir.name}.npz")
        for cand in candidates:
            if cand.is_file():
                return cand
    except Exception:
        return None
    return None


def _write_alignment_npz(
    path: Path,
    align: np.ndarray,
    base: Optional[np.ndarray],
    support: Optional[np.ndarray],
) -> None:
    payload = {"align_transform": align.astype(np.float32)}
    if base is not None:
        payload["base_transform"] = base.astype(np.float32)
    if support is not None:
        payload["support_transform"] = support.astype(np.float32)
    np.savez(path, **payload)


def _load_alignment_payload(
    path: Path,
) -> Optional[Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]]:
    try:
        data = np.load(path)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[warn] Failed to read alignment candidate {path}: {exc}")
        return None
    # Require align_transform to be present - no fallback reconstruction
    if "align_transform" not in data:
        print(f"[warn] Alignment file {path} missing 'align_transform'. Skipping.")
        return None

    align = data["align_transform"].astype(np.float32)
    base = data.get("base_transform")
    support = data.get("support_transform")

    return align, base, support


def _alignment_cosine(
    train_c2w: np.ndarray, eval_c2w: np.ndarray, align: np.ndarray
) -> float:
    sys.path.append("examples")
    from datasets.normalize import transform_cameras  # noqa: WPS433

    transformed = transform_cameras(align, eval_c2w.copy())

    def _mean_up(c2w: np.ndarray) -> np.ndarray:
        ups = c2w[:, :3, 1]
        ups = ups / np.linalg.norm(ups, axis=1, keepdims=True)
        mean = ups.mean(axis=0)
        return mean / np.linalg.norm(mean)

    train_up = _mean_up(train_c2w)
    eval_up = _mean_up(transformed)
    return float(np.dot(train_up, eval_up))


def _compute_alignment_via_parser(
    train_dir: Path,
    eval_dir: Path,
    train_every: int,
    eval_every: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    sys.path.append("examples")
    from datasets.colmap import Parser  # noqa: WPS433

    # Normalize both datasets independently so we can measure the relative
    # transform that maps eval cameras into the training coordinate frame.
    train_parser = Parser(
        data_dir=str(train_dir), factor=1, normalize=True, test_every=train_every
    )
    eval_parser = Parser(
        data_dir=str(eval_dir), factor=1, normalize=True, test_every=eval_every
    )

    train_normalization = train_parser.transform
    eval_normalization = eval_parser.transform
    align = train_normalization @ np.linalg.inv(eval_normalization)

    cosine = _alignment_cosine(train_parser.camtoworlds, eval_parser.camtoworlds, align)
    print(f"[verify] alignment cosine (train up · aligned eval up) = {cosine:.4f}")
    return (
        align.astype(np.float32),
        train_normalization.astype(np.float32),
        eval_normalization.astype(np.float32),
        cosine,
    )


def _ensure_alignment(
    train_dir: Optional[Path],
    eval_dir: Path,
    out_path: Path,
    train_every: int,
    eval_every: int,
) -> Path:
    """Fetch or compute train->eval alignment (standardized to train frame)."""
    if out_path.is_file():
        return out_path

    align = base = support = None
    cosine = None
    if train_dir is not None:
        try:
            align, base, support, cosine = _compute_alignment_via_parser(
                train_dir, eval_dir, train_every, eval_every
            )
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[warn] Failed to compute alignment via Parser: {exc}")
            align = base = support = None

    if align is None:
        shared = _shared_alignment_candidate(train_dir, eval_dir)
        if shared is not None and shared.is_file():
            payload = _load_alignment_payload(shared)
            if payload:
                align, base, support = payload

    if align is None:
        candidate = _dataset_alignment(train_dir or eval_dir, eval_dir)
        if candidate and candidate.is_file():
            payload = _load_alignment_payload(candidate)
            if payload:
                align, base, support = payload

    if align is None:
        raise RuntimeError(
            "Unable to resolve dataset alignment; supply --train_data_dir or --align_path explicitly."
        )

    if cosine is not None and cosine < 0.5:
        raise RuntimeError(
            f"Alignment verification failed (cosine={cosine:.3f}). "
            "Check dataset transforms for flipped poses."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_alignment_npz(out_path, align, base, support)
    return out_path


def _select_eval_frames(
    train_dir: Path,
    eval_dir: Path,
    k: int,
    train_every: int,
    eval_every: int,
    stride: int,
) -> List[str]:
    """Return up to k eval image names nearest to training cameras by center.

    If k <= 0, returns the full eval image list.
    """
    sys.path.append("examples")
    from datasets.colmap import Parser  # noqa: WPS433 (runtime import)

    train_parser = Parser(
        data_dir=str(train_dir), factor=1, normalize=False, test_every=train_every
    )
    eval_parser = Parser(
        data_dir=str(eval_dir), factor=1, normalize=False, test_every=eval_every
    )

    train_centers = train_parser.camtoworlds[:, :3, 3]
    eval_centers = eval_parser.camtoworlds[:, :3, 3]
    if train_centers.size == 0 or eval_centers.size == 0:
        raise RuntimeError("Insufficient cameras available to select eval subset.")

    dists = []
    for idx, name in enumerate(eval_parser.image_names):
        center = eval_centers[idx]
        min_dist = float(np.linalg.norm(train_centers - center, axis=1).min())
        dists.append((min_dist, name))
    dists.sort(key=lambda x: x[0])

    ordered = [name for _, name in dists]
    if stride > 1:
        ordered = ordered[::stride]

    if k <= 0:
        return ordered
    return ordered[: min(k, len(ordered))]


def main() -> int:
    parser = argparse.ArgumentParser(description="External eval-only render refresher")
    parser.add_argument(
        "--train_result_dir",
        required=True,
        type=Path,
        help="Training result dir (with ckpts/)",
    )
    parser.add_argument(
        "--eval_dir", required=True, type=Path, help="External COLMAP dataset directory"
    )
    parser.add_argument(
        "--train_data_dir",
        type=Path,
        default=None,
        help="Training COLMAP dataset directory (for alignment/closest selection)",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Output dir (defaults to <train_result_dir>/eval_on_external)",
    )
    parser.add_argument(
        "--k", type=int, default=10, help="Number of eval frames to render (0 = all)"
    )
    parser.add_argument(
        "--eval_stride",
        type=int,
        default=1,
        help="Stride for eval image selection (1 = use every frame)",
    )
    parser.add_argument(
        "--train_test_every",
        type=int,
        default=1,
        help="Subsample step for training cameras when selecting k frames",
    )
    parser.add_argument(
        "--eval_test_every",
        type=int,
        default=1,
        help="Subsample step for eval cameras and during eval",
    )
    parser.add_argument(
        "--use_covisible",
        action="store_true",
        help="Apply covisible masks for metrics (renders unmasked)",
    )
    parser.add_argument(
        "--covisible_dir",
        type=Path,
        default=None,
        help="Path to covisible root (expects <root>/val/<image>.png)",
    )
    parser.add_argument(
        "--align_path",
        type=Path,
        default=None,
        help="Optional precomputed alignment npz path",
    )
    parser.add_argument(
        "--render_video", action="store_true", help="Also render a video trajectory"
    )
    parser.add_argument(
        "--vis_concat_gt",
        action="store_true",
        default=False,
        help="If set, save GT|pred mosaics; otherwise save predictions only.",
    )
    parser.add_argument(
        "--python_bin",
        type=str,
        default=sys.executable or "python",
        help="Python executable to invoke",
    )
    parser.add_argument(
        "--dry_run", action="store_true", help="Print the planned command and exit"
    )

    args = parser.parse_args()

    train_result_dir = args.train_result_dir.resolve()
    eval_dir = args.eval_dir.resolve()
    out_dir = (args.out_dir or (train_result_dir / "eval_on_external")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    torch_ext_dir_env = os.environ.get("GSPLAT_TORCH_EXT_DIR")
    if torch_ext_dir_env:
        torch_ext_dir = Path(torch_ext_dir_env).expanduser().resolve()
    else:
        torch_ext_dir = (REPO_ROOT / ".torch_extensions").resolve()
    torch_ext_dir.mkdir(parents=True, exist_ok=True)

    # Resolve checkpoint
    ckpt = _latest_ckpt(train_result_dir)

    # Build a small eval list
    eval_list_path = out_dir / "selected_eval_images.txt"
    if args.eval_stride <= 0:
        raise ValueError("--eval_stride must be >= 1")

    if args.train_data_dir is not None:
        selected = _select_eval_frames(
            train_dir=args.train_data_dir.resolve(),
            eval_dir=eval_dir,
            k=args.k,
            train_every=args.train_test_every,
            eval_every=args.eval_test_every,
            stride=args.eval_stride,
        )
    else:
        # Fallback: select first K eval frames deterministically
        sys.path.append("examples")
        from datasets.colmap import Parser  # type: ignore

        eval_parser = Parser(
            data_dir=str(eval_dir),
            factor=1,
            normalize=False,
            test_every=args.eval_test_every,
        )
        names = list(eval_parser.image_names)
        if args.eval_stride > 1:
            names = names[:: args.eval_stride]
        selected = names if args.k <= 0 else names[: args.k]
    with eval_list_path.open("w", encoding="utf-8") as handle:
        for name in selected:
            handle.write(f"{name}\n")

    # Alignment: honor an explicitly provided path verbatim and only fall back to
    # auto-resolution when none was supplied.
    align_path: Optional[Path]
    if args.align_path is not None:
        align_path = Path(args.align_path).expanduser().resolve()
    elif args.train_data_dir is not None:
        candidate = out_dir / "alignments" / "eval_to_train.npz"
        try:
            align_path = _ensure_alignment(
                train_dir=args.train_data_dir.resolve(),
                eval_dir=eval_dir,
                out_path=candidate,
                train_every=args.train_test_every,
                eval_every=args.eval_test_every,
            )
        except Exception as exc:  # alignment is optional
            print(f"[warn] Failed to compute alignment: {exc}. Proceeding without.")
            align_path = None
    else:
        align_path = None

    # Optional covisible usage for metrics (renders will remain unmasked)
    covi_args: List[str] = []
    if args.use_covisible:
        if args.covisible_dir is None:
            print(
                "[warn] --use_covisible set but --covisible_dir is not provided; skipping masks."
            )
        else:
            covisible_root = args.covisible_dir.resolve()
            covi_args = [
                "--eval_use_covisible",
                "--eval_dycheck_metrics",
                "--eval_covisible_dir",
                str(covisible_root),
            ]

    # Build trainer command for eval-only
    cmd = [
        args.python_bin,
        "examples/simple_trainer.py",
        "default",
        "--disable_viewer",
        "--data_factor",
        "1",
        "--data_dir",
        str(eval_dir),
        "--result_dir",
        str(out_dir),
        "--ckpt",
        str(ckpt),
        "--test_every",
        str(args.eval_test_every),
        "--eval_list",
        str(eval_list_path),
    ]

    # Ensure saved renders have no GT pasted unless explicitly requested.
    if args.vis_concat_gt:
        cmd.append("--vis-concat-gt")
    else:
        cmd.append("--no-vis-concat-gt")

    if align_path is not None and align_path.is_file():
        cmd.extend(["--dataset-transform-path", str(align_path)])

    if not args.render_video:
        cmd.append("--disable_video")

    cmd.extend(covi_args)

    if _pose_opt_enabled(train_result_dir):
        print(
            "[info] Detected pose_opt=True in cfg.yml; enabling pose optimization during eval."
        )
        cmd.append("--pose_opt")

    print(
        json.dumps(
            {
                "ckpt": str(ckpt),
                "eval_dir": str(eval_dir),
                "out_dir": str(out_dir),
                "selected_count": len(selected),
                "python": args.python_bin,
                "command": " ".join(cmd),
            },
            indent=2,
        )
    )

    if args.dry_run:
        return 0

    # Launch eval-only
    env = os.environ.copy()
    pythonpath_entries = [str(REPO_ROOT)]
    if env.get("PYTHONPATH"):
        pythonpath_entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    env["TORCH_EXTENSIONS_DIR"] = str(torch_ext_dir)

    proc = subprocess.run(cmd, check=False, env=env)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
