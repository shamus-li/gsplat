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


def _load_rgb(path: str) -> np.ndarray:
    import imageio.v2 as imageio

    return imageio.imread(path)[..., :3]


def _resize_and_pad_image(
    image: np.ndarray,
    target_hw: Optional[Tuple[int, int]],
) -> np.ndarray:
    """Resize with aspect preservation and symmetric padding to target H×W."""
    if target_hw is None:
        return image

    target_h, target_w = target_hw
    if image.shape[0] == target_h and image.shape[1] == target_w:
        return image

    import cv2

    h, w = image.shape[:2]
    scale = min(target_h / h, target_w / w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left
    return cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=0,
    )


def _maybe_downscale_image(
    image: np.ndarray,
    max_hw: Optional[int],
) -> np.ndarray:
    """Uniformly downscale so the longer side <= max_hw (if provided)."""
    if max_hw is None or max_hw <= 0:
        return image

    h, w = image.shape[:2]
    longer = max(h, w)
    if longer <= max_hw:
        return image

    scale = max_hw / float(longer)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    import cv2

    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _empty_cache(device: torch.device) -> None:
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        with torch.cuda.device(idx):
            torch.cuda.empty_cache()


class ColmapRAFTDenseDataset(torch.utils.data.Dataset):
    """Lazy dataset that yields (base_img, support_img, base_idx, support_idx)."""

    def __init__(
        self,
        base_paths: Sequence[str],
        support_paths: Sequence[str],
        target_hw: Optional[Tuple[int, int]],
        max_hw: Optional[int],
    ):
        self.base_paths = list(base_paths)
        self.support_paths = list(support_paths)
        self.target_hw = target_hw
        self.max_hw = max_hw
        self.B = len(self.base_paths)
        self.S = len(self.support_paths)

    def __len__(self) -> int:
        return self.B * self.S

    def __getitem__(self, index: int):
        bi, si = divmod(index, self.S)
        base_img = _prepare_image(_load_rgb(self.base_paths[bi]), self.max_hw, self.target_hw)
        support_img = _prepare_image(_load_rgb(self.support_paths[si]), self.max_hw, self.target_hw)
        return base_img, support_img, bi, si


def _prepare_image(
    image: np.ndarray,
    max_hw: Optional[int],
    target_hw: Optional[Tuple[int, int]],
) -> np.ndarray:
    image = _maybe_downscale_image(image, max_hw)
    return _resize_and_pad_image(image, target_hw)


def _flow_to_warp(flow: np.ndarray) -> np.ndarray:
    """Compute warp coordinates from optical flow."""
    H, W = flow.shape[:2]
    x, y = np.meshgrid(
        np.arange(W, dtype=flow.dtype),
        np.arange(H, dtype=flow.dtype),
        indexing="xy",
    )
    grid = np.stack([x, y], axis=-1)
    return grid + flow


def compute_covisible_for_base(
    base_parser: ColmapParser,
    base_indices: Sequence[int],
    support_parser: ColmapParser,
    support_indices: Sequence[int],
    *,
    factor: int,
    out_dir: Path,
    metadata_root: Path,
    batch_size: int = 32,
    num_workers: int = 4,
    device: torch.device = torch.device("cuda"),
    num_min_frames: int = 5,
    min_frame_ratio: float = 0.1,
    max_hw: Optional[int] = None,
) -> None:
    path_ops.mkdir(out_dir.as_posix())
    metadata_root.mkdir(parents=True, exist_ok=True)

    base_paths = [base_parser.image_paths[i] for i in base_indices]
    support_paths = [support_parser.image_paths[i] for i in support_indices]

    if not base_paths:
        raise ValueError(
            "[covisible] Resolved zero base images. "
            f"Dataset='{getattr(base_parser, 'data_dir', '?')}', "
            f"indices={len(base_indices)}, "
            f"test_every={getattr(base_parser, 'test_every', '?')}."
        )

    if not support_paths:
        raise ValueError(
            "[covisible] Resolved zero support images. "
            f"Dataset='{getattr(support_parser, 'data_dir', '?')}', "
            f"indices={len(support_indices)}, "
            f"test_every={getattr(support_parser, 'test_every', '?')}."
        )

    # Determine target HW from the first support image.
    support_preview = _maybe_downscale_image(_load_rgb(support_paths[0]), max_hw)
    target_hw = support_preview.shape[:2]

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

    dataset = ColmapRAFTDenseDataset(
        base_paths=base_paths,
        support_paths=support_paths,
        target_hw=target_hw,
        max_hw=max_hw,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    if device.type == "cuda":
        torch.cuda.set_device(device.index if device.index is not None else 0)
    model = get_raft().to(device).eval()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    S = len(support_paths)
    B = len(base_paths)

    min_ratio_visible = max(1, int(np.ceil(min_frame_ratio * S)))
    if S < num_min_frames:
        visible_thresh = min(S, min_ratio_visible)
    else:
        visible_thresh = min(S, max(num_min_frames, min_ratio_visible))
    if visible_thresh <= 0:
        raise ValueError(
            f"Computed non-positive visibility threshold ({visible_thresh})."
            " Check num_min_frames/min_frame_ratio settings."
        )
    print(
        f"[covisible] Support frames: {S}, keeping pixels visible in >= {visible_thresh} frames.",
        flush=True,
    )

    occ_cache: List[List[np.ndarray]] = [[] for _ in range(B)]
    completed = 0

    import cv2

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Computing covisible"):
            img0_batch, img1_batch, bi_batch, _ = batch
            img0_t = (
                img0_batch.permute(0, 3, 1, 2)
                .contiguous()
                .float()
                .to(device, non_blocking=True)
            )
            img1_t = (
                img1_batch.permute(0, 3, 1, 2)
                .contiguous()
                .float()
                .to(device, non_blocking=True)
            )

            padder = InputPadder(img0_t.shape)
            img0_t, img1_t = padder.pad(img0_t, img1_t)

            flow_fw = model(img0_t, img1_t, iters=20, test_mode=True)[1]
            flow_bw = model(img1_t, img0_t, iters=20, test_mode=True)[1]

            flow_fw = (
                padder.unpad(flow_fw).permute(0, 2, 3, 1).cpu().numpy()
            )
            flow_bw = (
                padder.unpad(flow_bw).permute(0, 2, 3, 1).cpu().numpy()
            )

            bi_list = bi_batch.tolist()
            for k in range(flow_fw.shape[0]):
                fw = flow_fw[k]
                bw = flow_bw[k]
                warp = _flow_to_warp(fw)
                bw_res = cv2.remap(
                    bw, warp[..., 0], warp[..., 1], cv2.INTER_LINEAR
                )
                fb_sq_diff = np.sum((fw + bw_res) ** 2, axis=-1, keepdims=True)
                fb_sum_sq = np.sum(fw**2 + bw_res**2, axis=-1, keepdims=True)
                occ = (fb_sq_diff > 0.01 * fb_sum_sq + 0.5).astype(np.float32)

                bi = int(bi_list[k])
                occ_cache[bi].append(occ)
                if len(occ_cache[bi]) == S:
                    support_occs = np.stack(occ_cache[bi], axis=0)
                    visible_counts = (1.0 - support_occs).sum(axis=0)
                    covisible = (visible_counts >= visible_thresh).astype(np.float32)
                    if not np.any(covisible):
                        print(
                            "[covisible] Empty mask for",
                            base_parser.image_names[base_indices[bi]],
                            "-> defaulting to full visibility",
                            flush=True,
                        )
                        covisible[:] = 1.0

                    rel_name = Path(base_parser.image_names[base_indices[bi]])
                    out_path = out_dir / rel_name.with_suffix(".png")
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    import imageio.v2 as imageio

                    mask = (covisible[..., 0] * 255).astype(np.uint8)
                    imageio.imwrite(out_path.as_posix(), mask)

                    occ_cache[bi] = []
                    completed += 1

            _empty_cache(device)

    if completed != B:
        missing = [base_parser.image_names[base_indices[i]] for i in range(B) if occ_cache[i]]
        raise RuntimeError(
            f"[covisible] Did not finish all base images. Pending masks for: {missing}"
        )


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
    ap.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size (number of image pairs) processed per RAFT step.",
    )
    ap.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader workers for streaming image IO.",
    )
    ap.add_argument("--num_min_frames", type=int, default=3)
    ap.add_argument("--min_frame_ratio", type=float, default=0.05)
    ap.add_argument(
        "--max_hw",
        type=int,
        default=640,
        help="Clamp images so the longer side is at most this many pixels before RAFT (0 disables).",
    )
    args = ap.parse_args()

    base_dir = Path(args.base_dir).resolve()
    support_dir = Path(args.support_dir).resolve() if args.support_dir else base_dir

    # Build parsers with normalization so we can persist alignment metadata.
    base_parser = ColmapParser(
        data_dir=str(base_dir), factor=args.factor, normalize=True, test_every=args.test_every
    )
    # Support cadence defaults to the base cadence unless explicitly overridden.
    support_test_every = args.support_test_every or args.test_every
    support_parser = ColmapParser(
        data_dir=str(support_dir), factor=args.factor, normalize=True, test_every=support_test_every
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
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        num_min_frames=args.num_min_frames,
        min_frame_ratio=args.min_frame_ratio,
        max_hw=args.max_hw,
    )


if __name__ == "__main__":
    main()
