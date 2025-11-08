#!/usr/bin/env python3
"""
Compute covisible masks for a COLMAP dataset using RAFT (from dycheck).

The masks are saved under:
    <base_dir>/covisible/<factor>x/<base_split>/<image_name>.png

Each mask highlights pixels that are co-visible across a set of support images
according to a forward-backward RAFT occlusion consistency check, following the
logic in dycheck/tools/process_covisible.py.

Require: dycheck repository available at ../dycheck relative to this script,
or otherwise importable in PYTHONPATH, with its submodule RAFT ready.
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

try:
    # Allow running from gsplat/examples with sibling dycheck
    DY_ROOT = Path(__file__).resolve().parents[2] / "dycheck"
    if DY_ROOT.is_dir():
        import sys

        sys.path.insert(0, str(DY_ROOT))
    from dycheck.processors.raft import (
        InputPadder,  # type: ignore
        compute_chunk_raft_flow,
        compute_raft_flow,
        get_raft,
    )
    from dycheck.utils import path_ops
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Failed to import dycheck RAFT modules. Ensure '../dycheck' exists and submodules are initialized."
    ) from e

from datasets.colmap import Dataset as ColmapDataset
from datasets.colmap import Parser as ColmapParser


def _select_indices(
    parser: ColmapParser,
    split: str,
    test_every: int,
    match_string: Optional[str] = None,
) -> List[int]:
    """Mimic examples/datasets/colmap.Dataset split and optional filtering.

    Returns a list of indices into parser.image_names.
    """
    import re

    indices = np.arange(len(parser.image_names))
    if test_every <= 1:
        source = indices
    elif split == "train":
        source = indices[indices % test_every != 0]
    elif split == "val":
        source = indices[indices % test_every == 0]
    else:
        raise ValueError(f"Unsupported split '{split}'. Use 'train' or 'val'.")

    if match_string is None or match_string == "":
        return list(source.tolist())

    token_pattern = re.compile(
        rf"(?<![A-Za-z0-9]){re.escape(match_string)}(?![A-Za-z0-9])",
        flags=re.IGNORECASE,
    )
    filtered = [
        i for i in source if token_pattern.search(parser.image_names[i])
    ]
    if not filtered:
        raise ValueError(
            f"No images matched match_string '{match_string}' within split '{split}'."
        )
    return filtered


def _load_rgbs(paths: Sequence[str]) -> np.ndarray:
    import imageio.v2 as imageio

    imgs = [imageio.imread(p)[..., :3] for p in paths]
    return np.stack(imgs, axis=0)


def _resize_and_pad_stack(
    images: np.ndarray,
    target_hw: Tuple[int, int],
) -> np.ndarray:
    """Resize with aspect preservation and symmetric padding to target H×W."""
    target_h, target_w = target_hw
    if images.shape[1] == target_h and images.shape[2] == target_w:
        return images

    import cv2

    resized_padded = []
    for img in images:
        h, w = img.shape[:2]
        scale = min(target_h / h, target_w / w)
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        pad_top = (target_h - new_h) // 2
        pad_bottom = target_h - new_h - pad_top
        pad_left = (target_w - new_w) // 2
        pad_right = target_w - new_w - pad_left
        padded = cv2.copyMakeBorder(
            resized,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=0,
        )
        resized_padded.append(padded)

    return np.stack(resized_padded, axis=0)


def _empty_cache(device: torch.device) -> None:
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        with torch.cuda.device(idx):
            torch.cuda.empty_cache()


def compute_covisible_for_base(
    base_parser: ColmapParser,
    base_indices: Sequence[int],
    support_parser: ColmapParser,
    support_indices: Sequence[int],
    *,
    factor: int,
    out_dir: Path,
    metadata_root: Path,
    chunk: int = 64,
    micro_chunk: Optional[int] = None,
    device: torch.device = torch.device("cuda"),
    num_min_frames: int = 5,
    min_frame_ratio: float = 0.1,
) -> None:
    # Ensure output dirs exist
    path_ops.mkdir(out_dir.as_posix())
    metadata_root.mkdir(parents=True, exist_ok=True)

    # Collect image paths
    base_paths = [base_parser.image_paths[i] for i in base_indices]
    support_paths = [support_parser.image_paths[i] for i in support_indices]

    if len(base_paths) == 0:
        raise ValueError(
            "[covisible] Resolved zero base images. "
            f"Dataset='{getattr(base_parser, 'data_dir', '?')}', "
            f"indices={len(base_indices)}, "
            f"test_every={getattr(base_parser, 'test_every', '?')}."
        )

    if len(support_paths) == 0:
        raise ValueError(
            "[covisible] Resolved zero support images. "
            f"Dataset='{getattr(support_parser, 'data_dir', '?')}', "
            f"indices={len(support_indices)}, "
            f"test_every={getattr(support_parser, 'test_every', '?')}."
        )

    # Load images as float32 [0,255] expected by RAFT utils
    base_rgbs = _load_rgbs(base_paths)
    support_rgbs = _load_rgbs(support_paths)
    support_rgbs = _resize_and_pad_stack(support_rgbs, base_rgbs.shape[1:3])

    # Persist alignment information between base and support normalizations.
    base_transform = base_parser.transform.astype(np.float64)
    support_transform = support_parser.transform.astype(np.float64)
    align_transform = support_transform @ np.linalg.inv(base_transform)
    np.savez(
        metadata_root / "alignment.npz",
        base_transform=base_transform.astype(np.float32),
        support_transform=support_transform.astype(np.float32),
        align_transform=align_transform.astype(np.float32),
        factor=np.array([factor], dtype=np.int32),
        base_split=np.array([base_parser.test_every], dtype=np.int32),
    )

    # Prepare RAFT model
    if device.type == "cuda":
        torch.cuda.set_device(device.index if device.index is not None else 0)
    model = get_raft().to(device).eval()

    # For memory, process in chunks over support dimension
    S = len(support_rgbs)
    B = len(base_rgbs)

    max_micro = max(1, micro_chunk if micro_chunk is not None else chunk)

    # Iterate base images, compute occlusion against all support frames
    for bi in tqdm(range(B), desc="Dumping covisible"):
        cache: List[np.ndarray] = []  # occlusion masks for this base
        img0 = base_rgbs[bi]

        # Chunk over support images
        for start in range(0, S, chunk):
            end = min(start + chunk, S)
            imgs1_chunk = support_rgbs[start:end]

            remaining = end - start
            sub_offset = 0
            current_cap = min(max_micro, remaining)

            while sub_offset < remaining:
                attempt = min(current_cap, remaining - sub_offset)
                success = False
                while not success:
                    try:
                        imgs1 = imgs1_chunk[sub_offset : sub_offset + attempt]

                        with torch.inference_mode():
                            img0_b = np.repeat(img0[None], repeats=attempt, axis=0)

                            img0_t = (
                                torch.from_numpy(img0_b.astype(np.uint8))
                                .permute(0, 3, 1, 2)
                                .float()
                                .to(device, non_blocking=True)
                            )
                            img1_t = (
                                torch.from_numpy(imgs1.astype(np.uint8))
                                .permute(0, 3, 1, 2)
                                .float()
                                .to(device, non_blocking=True)
                            )
                            padder = InputPadder(img0_t.shape)
                            img0_t, img1_t = padder.pad(img0_t, img1_t)

                            flow_fw = model(img0_t, img1_t, iters=20, test_mode=True)[
                                1
                            ]
                            flow_bw = model(img1_t, img0_t, iters=20, test_mode=True)[
                                1
                            ]
                            flow_fw = (
                                padder.unpad(flow_fw)
                                .permute(0, 2, 3, 1)
                                .cpu()
                                .numpy()
                            )
                            flow_bw = (
                                padder.unpad(flow_bw)
                                .permute(0, 2, 3, 1)
                                .cpu()
                                .numpy()
                            )

                        occs: List[np.ndarray] = []
                        import cv2

                        def flow_to_warp(flow: np.ndarray) -> np.ndarray:
                            H, W = flow.shape[:2]
                            x, y = np.meshgrid(
                                np.arange(W, dtype=flow.dtype),
                                np.arange(H, dtype=flow.dtype),
                                indexing="xy",
                            )
                            grid = np.stack([x, y], axis=-1)
                            return grid + flow

                        for k in range(flow_fw.shape[0]):
                            fw = flow_fw[k]
                            bw = flow_bw[k]
                            warp = flow_to_warp(fw)
                            bw_res = cv2.remap(
                                bw, warp[..., 0], warp[..., 1], cv2.INTER_LINEAR
                            )
                            fb_sq_diff = np.sum(
                                (fw + bw_res) ** 2, axis=-1, keepdims=True
                            )
                            fb_sum_sq = np.sum(
                                fw**2 + bw_res**2, axis=-1, keepdims=True
                            )
                            occ = (fb_sq_diff > 0.01 * fb_sum_sq + 0.5).astype(
                                np.float32
                            )
                            occs.append(occ)

                        cache.extend(occs)
                        success = True
                        _empty_cache(device)
                    except RuntimeError as e:
                        error_msg = str(e).lower()
                        is_oom = "out of memory" in error_msg or "integer out of range" in error_msg
                        if is_oom and attempt > 1:
                            new_attempt = max(1, attempt // 2)
                            if new_attempt != attempt:
                                print(
                                    f"[covisible] CUDA OOM detected ({str(e)[:50]}...); retrying with sub-batch size {new_attempt}",
                                    flush=True,
                                )
                            attempt = new_attempt
                            _empty_cache(device)
                            continue
                        raise

                sub_offset += attempt
                current_cap = attempt

        support_occs = np.array(cache, dtype=np.float32)  # [S, H, W, 1]
        # Determine pixels occluded by most supports; keep pixels visible in
        # at least T frames, same threshold rule as dycheck
        thresh = max(num_min_frames, int(min_frame_ratio * S))
        covisible = 1.0 - (
            (1.0 - support_occs).sum(axis=0) <= thresh
        ).astype(np.float32)

        # Save mask PNG aligned with base parser image name
        rel_name = Path(base_parser.image_names[base_indices[bi]])
        out_path = out_dir / rel_name.with_suffix(".png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        import imageio.v2 as imageio

        mask = (covisible[..., 0] * 255).astype(np.uint8)
        imageio.imwrite(out_path.as_posix(), mask)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, required=True, help="COLMAP dataset root (images/ and sparse/).")
    ap.add_argument("--support_dir", type=str, default=None, help="Support dataset root; defaults to base_dir.")
    ap.add_argument("--factor", type=int, default=1)
    ap.add_argument("--test_every", type=int, default=8, help="Train/val splitting cadence for base_dir.")
    ap.add_argument("--base_split", type=str, default="val", choices=["train", "val"])
    ap.add_argument("--base_match", type=str, default=None, help="Optional regex-safe token to filter base images.")
    ap.add_argument("--support_split", type=str, default="train", choices=["train", "val"], help="Support split (usually 'train').")
    ap.add_argument(
        "--support_test_every",
        type=int,
        default=None,
        help="Optional cadence for support split; defaults to --test_every.",
    )
    ap.add_argument(
        "--micro_chunk",
        type=int,
        default=None,
        help="Maximum number of support frames processed per RAFT call (defaults to --chunk).",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="CUDA device to run RAFT on (e.g., 'cuda:0').",
    )
    ap.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to write covisible masks and alignment metadata (defaults to <base_dir>/covisible).",
    )
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--num_min_frames", type=int, default=5)
    ap.add_argument("--min_frame_ratio", type=float, default=0.1)
    args = ap.parse_args()

    base_dir = Path(args.base_dir).resolve()
    support_dir = Path(args.support_dir).resolve() if args.support_dir else base_dir

    # Build parsers (normalize False; we only need images)
    base_parser = ColmapParser(
        data_dir=str(base_dir), factor=args.factor, normalize=False, test_every=args.test_every
    )
    # Support cadence defaults to the base cadence unless explicitly overridden.
    support_test_every = args.support_test_every or args.test_every
    support_parser = ColmapParser(
        data_dir=str(support_dir), factor=args.factor, normalize=False, test_every=support_test_every
    )

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError(f"Device '{args.device}' is not a CUDA device; RAFT requires CUDA.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available.")
    if torch.cuda.device_count() == 0:
        raise RuntimeError("No CUDA devices detected.")

    if device.index is not None:
        if device.index >= torch.cuda.device_count():
            raise ValueError(
                f"Requested device index {device.index}, but only {torch.cuda.device_count()} CUDA devices are visible."
            )
        torch.cuda.set_device(device.index)
        device = torch.device(f"cuda:{device.index}")
    else:
        default_idx = 0
        torch.cuda.set_device(default_idx)
        device = torch.device(f"cuda:{default_idx}")

    base_indices = _select_indices(
        base_parser, split=args.base_split, test_every=args.test_every, match_string=args.base_match
    )
    support_indices = _select_indices(
        support_parser, split=args.support_split, test_every=support_test_every, match_string=None
    )

    output_root = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else base_dir / "covisible"
    )

    out_dir = output_root / f"{args.factor}x" / args.base_split
    compute_covisible_for_base(
        base_parser,
        base_indices,
        support_parser,
        support_indices,
        factor=args.factor,
        out_dir=out_dir,
        metadata_root=output_root,
        chunk=args.chunk,
        micro_chunk=args.micro_chunk,
        device=device,
        num_min_frames=args.num_min_frames,
        min_frame_ratio=args.min_frame_ratio,
    )


if __name__ == "__main__":
    main()
