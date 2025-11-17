#!/usr/bin/env python3
"""Prepare shared static datasets for multi-camera few-image training.

This helper:
1. Validates a scene directory under ../gs7/dataset/<scene>/.
2. Builds staged subsets for monocular/iphone/stereo/lightfield + iphone_eval.
3. Runs VGGT/BA on the combined images (if needed) via preprocess_shared_gsplat utilities.
4. Splits the shared reconstruction back into per-subset datasets and writes metadata.
5. Emits a sentinel (.prep_complete) so repeated runs are skipped unless inputs change.

It is intended to be invoked by examples/static_preprocessing.slurm on an H100 node.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

_EXAMPLES_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EXAMPLES_DIR.parent
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
sys.path.append(str(_SCRIPTS_DIR))

import preprocess_for_gsplat as base_prep  # noqa: E402
import preprocess_shared_gsplat as shared  # noqa: E402

IMAGE_EXTS = getattr(
    base_prep,
    "IMAGE_EXTS",
    {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".ppm", ".pgm"},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, required=True, help="Path to the scene root (../gs7/dataset/<scene>).")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory to store the shared dataset (default: <scene>/static/shared).")
    parser.add_argument("--lf-calib", type=Path, default=None, help="Calibration tarball for LFR decoding (overrides auto-detect).")
    parser.add_argument("--lf-inner", type=int, default=2, help="Inner crop forwarded to process_lfr (default: 2).")
    parser.add_argument("--lf-downscale", type=float, default=None, help="Optional downscale forwarded to process_lfr.")
    parser.add_argument("--lf-to-colmap", type=Path, default=None, help="Path to lf_to_colmap.py.")
    parser.add_argument("--conda-exe", type=str, default=os.environ.get("CONDA_EXE", "conda"), help="Conda executable for VGGT.")
    parser.add_argument("--conda-env", type=str, default="transformers", help="Conda env name for VGGT.")
    parser.add_argument("--vggt-script", type=Path, default=base_prep.default_vggt_script(), help="Path to VGGT demo_colmap.py.")
    parser.add_argument("--stage", choices=("both", "vggt", "ba"), default="both", help="VGGT stage to run.")
    parser.add_argument("--stage-cache", type=str, default=None, help="Optional cache directory for split VGGT stages.")
    parser.add_argument("--skip-vggt", action="store_true", help="Skip VGGT if sparse/ already exists.")
    parser.add_argument("--copy-mode", choices=("symlink", "copy", "hardlink"), default="symlink", help="How to materialize combined/subset images.")
    parser.add_argument("--force", action="store_true", help="Rebuild even if .prep_complete exists.")
    return parser.parse_args()


def _ensure_exists(path: Path, kind: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Expected {kind} at {path}")
    return path


def _collect_eval_frames(eval_images: Path) -> List[Path]:
    frames = sorted(p for p in eval_images.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not frames:
        raise FileNotFoundError(f"No evaluation frames found under {eval_images}")
    return frames


def _relative_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    rel = os.path.relpath(src, dst.parent)
    dst.symlink_to(rel)


def _stage_subset(
    staging_root: Path,
    subset_name: str,
    files: Sequence[Path],
    base_dir: Path,
    prefix: str,
    kind: str,
) -> shared.SubsetConfig:
    subset_dir = staging_root / subset_name
    images_dir = subset_dir / "images"
    if subset_dir.exists():
        shutil.rmtree(subset_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    records: List[shared.ImageRecord] = []
    for file_path in files:
        rel = file_path.relative_to(base_dir)
        dest = images_dir / rel
        _relative_symlink(file_path, dest)
        flattened = rel.as_posix().replace("/", "__")
        combined_name = f"{prefix}__{flattened}"
        records.append(
            shared.ImageRecord(
                source_path=dest,
                relative_path=rel,
                combined_name=combined_name,
                subset=subset_name,
            )
        )
    subset = shared.SubsetConfig(name=subset_name, source_dir=subset_dir, prefix=prefix, kind=kind)
    subset.records = records
    print(f"Prepared subset {subset_name}: {len(records)} image(s) (prefix={prefix})")
    return subset


def _find_lfr(static_dir: Path) -> Path:
    candidates = sorted(static_dir.glob("*.LFR"))
    if not candidates:
        raise FileNotFoundError(f"No .LFR files found under {static_dir}")
    if len(candidates) > 1:
        print(f"WARNING: Multiple .LFR files found; using {candidates[0].name}")
    return candidates[0]


def _decode_lfr(
    static_dir: Path,
    lfr_file: Path,
    *,
    lf_inner: int,
    lf_downscale: float | None,
    lf_to_colmap: Path,
    lf_calib: Path,
    force: bool,
) -> Path:
    decode_root = static_dir / "lightfield_decoded"
    inner_dir = decode_root / f"inner_{lf_inner:02d}"
    images_dir = inner_dir / "images"
    needs_decode = force or not (images_dir.exists() and any(images_dir.glob("*")))
    if needs_decode:
        task = base_prep.Task(kind="lfr", source=lfr_file, label=lfr_file.stem)
        print(f"Decoding LFR {lfr_file.name} -> {inner_dir}")
        base_prep.ensure_ready_dir(decode_root, overwrite=False)
        base_prep.process_lfr(
            task=task,
            out_dir=decode_root,
            lf_script=lf_to_colmap,
            calib_tar=lf_calib,
            inner=lf_inner,
            downscale=lf_downscale,
        )
    if not images_dir.exists():
        raise RuntimeError(f"LFR decode failed; missing {images_dir}")
    lf_images = sorted(p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not lf_images:
        raise RuntimeError(f"No decoded lightfield images found under {images_dir}")
    print(f"Lightfield subset: {len(lf_images)} decoded view(s)")
    return images_dir


def _gather_static_files(static_dir: Path) -> Dict[str, Path]:
    required = ["wide.png", "tele.png", "ultrawide.png", "stereo_left.png", "stereo_right.png"]
    files: Dict[str, Path] = {}
    for name in required:
        files[f"static/{name}"] = _ensure_exists(static_dir / name, f"static image ({name})")
    files["static/LFR"] = _find_lfr(static_dir)
    return files


def _file_stats(paths: Dict[str, Path]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for key, path in paths.items():
        info = path.stat()
        stats[key] = {"mtime": info.st_mtime, "size": info.st_size}
    return stats


def _sources_changed(prev: Dict[str, Dict[str, float]], current: Dict[str, Dict[str, float]]) -> bool:
    prev_keys = set(prev.keys())
    curr_keys = set(current.keys())
    if prev_keys != curr_keys:
        return True
    for key in curr_keys:
        if prev[key] != current[key]:
            return True
    return False


def _dir_signature(root: Path) -> str | None:
    if not root.exists():
        return None
    entries: List[str] = []
    for file_path in sorted(root.rglob("*")):
        if not file_path.is_file():
            continue
        rel = file_path.relative_to(root).as_posix()
        info = file_path.stat()
        entries.append(f"{rel}:{info.st_size}:{info.st_mtime}")
    if not entries:
        return None
    payload = "|".join(entries).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_prepared_dataset(args: argparse.Namespace) -> None:
    scene_dir = args.scene_dir.expanduser().resolve()
    static_dir = _ensure_exists(scene_dir / "static", "static directory")
    eval_dir = _ensure_exists(scene_dir / "iphone-eval", "iphone-eval directory")
    eval_images_dir = _ensure_exists(eval_dir / "images", "iphone-eval/images")

    output_dir = (args.output_dir.expanduser().resolve() if args.output_dir else (static_dir / "shared")).resolve()
    sentinel_path = output_dir / ".prep_complete"
    staging_root = output_dir / ".staging"

    lf_to_colmap = (args.lf_to_colmap.expanduser().resolve() if args.lf_to_colmap else base_prep.default_lf_script(base_prep.default_repo_root()))
    if not lf_to_colmap.exists():
        raise FileNotFoundError(f"lf_to_colmap script not found: {lf_to_colmap}")

    lf_calib = (args.lf_calib.expanduser().resolve() if args.lf_calib else base_prep.infer_calibration(base_prep.default_repo_root()))
    if lf_calib is None or not lf_calib.exists():
        raise FileNotFoundError("Calibration tarball not found; specify --lf-calib.")

    static_files = _gather_static_files(static_dir)
    eval_frames = _collect_eval_frames(eval_images_dir)
    source_map: Dict[str, Path] = {**static_files}
    for frame in eval_frames:
        rel = eval_images_dir.relative_to(scene_dir) / frame.name
        source_map[str(rel)] = frame

    current_stats = _file_stats(source_map)

    if sentinel_path.exists() and not args.force:
        with sentinel_path.open("r") as f:
            prev = json.load(f)
        prev_sources = prev.get("sources", {})
        if not _sources_changed(prev_sources, current_stats):
            print(f"Shared dataset already prepared at {output_dir} (no changes detected). Use --force to rebuild.")
            return
        print("Detected changes in source files since last preprocessing; rebuilding dataset...")

    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    lfr_file = static_files["static/LFR"]
    decoded_images_root = _decode_lfr(
        static_dir,
        lfr_file,
        lf_inner=args.lf_inner,
        lf_downscale=args.lf_downscale,
        lf_to_colmap=lf_to_colmap,
        lf_calib=lf_calib,
        force=args.force,
    )

    subsets: List[shared.SubsetConfig] = []
    subsets.append(_stage_subset(staging_root, "monocular", [static_dir / "wide.png"], static_dir, "mono", "train"))
    subsets.append(
        _stage_subset(
            staging_root,
            "iphone",
            [static_dir / "wide.png", static_dir / "tele.png", static_dir / "ultrawide.png"],
            static_dir,
            "iphone",
            "train",
        )
    )
    subsets.append(
        _stage_subset(
            staging_root,
            "stereo",
            [static_dir / "stereo_left.png", static_dir / "stereo_right.png"],
            static_dir,
            "stereo",
            "train",
        )
    )
    lf_images = sorted(p for p in decoded_images_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    subsets.append(_stage_subset(staging_root, "lightfield", lf_images, decoded_images_root, "lf", "train"))
    subsets.append(_stage_subset(staging_root, "iphone_eval", eval_frames, eval_images_dir, "eval", "eval"))

    mapping = shared.prepare_combined_images(output_dir=output_dir, subsets=subsets, copy_mode=args.copy_mode, overwrite=True)

    if not args.skip_vggt:
        shared.run_vggt_reconstruction(
            dataset_dir=output_dir,
            conda_exe=args.conda_exe,
            conda_env=args.conda_env,
            vggt_script=args.vggt_script.expanduser().resolve(),
            stage=args.stage,
            stage_cache=args.stage_cache,
            overwrite=args.force,
        )
    else:
        print("Skipping VGGT (--skip-vggt).")

    sparse_dir = output_dir / "sparse"
    if not sparse_dir.exists():
        raise RuntimeError(f"Shared sparse directory not found at {sparse_dir}")

    with tempfile.TemporaryDirectory(prefix="colmap_txt_") as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        combined_txt = tmpdir / "combined_txt"
        shared._convert_sparse_to_text(sparse_dir, combined_txt)
        shared.materialize_subset_dirs(
            output_dir=output_dir,
            subsets=subsets,
            copy_mode=args.copy_mode,
            combined_sparse=sparse_dir,
            combined_txt=combined_txt,
        )

    shared.write_split_lists(output_dir, subsets)
    shared.write_metadata(
        output_dir=output_dir,
        subsets=subsets,
        combined_mapping=mapping,
        train_test_every=1,
        eval_test_every=1,
    )

    sentinel_payload = {
        "scene": scene_dir.name,
        "timestamp": os.path.getmtime(sparse_dir),
        "config": {
            "lf_inner": args.lf_inner,
            "stage": args.stage,
            "copy_mode": args.copy_mode,
            "skip_vggt": args.skip_vggt,
        },
        "sources": current_stats,
        "subsets": {subset.name: len(subset.records) for subset in subsets},
        "lightfield_decode_signature": _dir_signature(decoded_images_root),
    }
    with sentinel_path.open("w") as f:
        json.dump(sentinel_payload, f, indent=2)
    print(f"Wrote sentinel to {sentinel_path}")


def main() -> None:
    args = parse_args()
    build_prepared_dataset(args)


if __name__ == "__main__":
    main()
