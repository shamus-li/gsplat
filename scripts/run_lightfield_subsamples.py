#!/usr/bin/env python3
"""Launch multiple 3DGS training runs with different light-field subsamples."""

from __future__ import annotations

import argparse
import random
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Sequence


@dataclass(frozen=True)
class ImageView:
    """Metadata for a single light-field image."""

    name: str
    row: int
    col: int
    row_idx: int
    col_idx: int


@dataclass(frozen=True)
class SubsampleContext:
    """Reusable context describing the light-field grid."""

    rows: Sequence[int]
    cols: Sequence[int]

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def col_count(self) -> int:
        return len(self.cols)


Selector = Callable[[List[ImageView], SubsampleContext, random.Random], List[ImageView]]


@dataclass(frozen=True)
class SubsampleSpec:
    """A named subsampling strategy."""

    name: str
    description: str
    selector: Selector


def parse_image_grid(image_dir: Path) -> tuple[List[ImageView], SubsampleContext]:
    """Parse filenames in a COLMAP light-field folder into a grid representation."""
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory '{image_dir}' does not exist.")

    valid_suffixes = {".png", ".jpg", ".jpeg"}
    files = sorted(
        [p for p in image_dir.iterdir() if p.suffix.lower() in valid_suffixes]
    )
    if not files:
        raise ValueError(f"No image files found in '{image_dir}'.")

    rows_found = set()
    cols_found = set()
    components: List[tuple[str, int, int]] = []

    for file in files:
        stem = file.stem
        parts = stem.split("_")
        if len(parts) < 2:
            raise ValueError(
                f"Expected filenames in 'row_col.ext' format, but got '{file.name}'."
            )
        try:
            row_val = int(parts[0])
            col_val = int(parts[1])
        except ValueError as exc:
            raise ValueError(
                f"Could not parse row/column indices from '{file.name}'."
            ) from exc
        rows_found.add(row_val)
        cols_found.add(col_val)
        components.append((file.name, row_val, col_val))

    rows = sorted(rows_found)
    cols = sorted(cols_found)
    row_index = {value: idx for idx, value in enumerate(rows)}
    col_index = {value: idx for idx, value in enumerate(cols)}

    images = [
        ImageView(
            name=name,
            row=row,
            col=col,
            row_idx=row_index[row],
            col_idx=col_index[col],
        )
        for name, row, col in components
    ]
    images.sort(key=lambda view: (view.row_idx, view.col_idx))

    return images, SubsampleContext(rows=rows, cols=cols)


def build_subsample_specs(random_count: int) -> List[SubsampleSpec]:
    """Return the set of subsampling strategies to evaluate."""

    def select_full(
        images: List[ImageView],
        _ctx: SubsampleContext,
        _rng: random.Random,
    ) -> List[ImageView]:
        return list(images)

    def select_center_cross(
        images: List[ImageView],
        ctx: SubsampleContext,
        _rng: random.Random,
    ) -> List[ImageView]:
        mid_row = ctx.row_count // 2
        mid_col = ctx.col_count // 2
        return [
            view
            for view in images
            if view.row_idx == mid_row or view.col_idx == mid_col
        ]

    def select_outer_ring(
        images: List[ImageView],
        ctx: SubsampleContext,
        _rng: random.Random,
    ) -> List[ImageView]:
        max_row = ctx.row_count - 1
        max_col = ctx.col_count - 1
        return [
            view
            for view in images
            if view.row_idx in (0, max_row) or view.col_idx in (0, max_col)
        ]

    def select_even_grid(
        images: List[ImageView],
        _ctx: SubsampleContext,
        _rng: random.Random,
    ) -> List[ImageView]:
        return [
            view
            for view in images
            if view.row_idx % 2 == 0 and view.col_idx % 2 == 0
        ]

    def select_checkerboard(
        images: List[ImageView],
        _ctx: SubsampleContext,
        _rng: random.Random,
    ) -> List[ImageView]:
        if not images:
            return []
        parity = (images[0].row_idx + images[0].col_idx) % 2
        return [
            view
            for view in images
            if (view.row_idx + view.col_idx) % 2 == parity
        ]

    def make_random_selector(count: int) -> Selector:
        def _selector(
            images: List[ImageView],
            _ctx: SubsampleContext,
            rng: random.Random,
        ) -> List[ImageView]:
            if not images or count <= 0:
                return []
            sample_size = min(count, len(images))
            selection = rng.sample(images, sample_size)
            selection.sort(key=lambda view: (view.row_idx, view.col_idx))
            return selection

        return _selector

    return [
        SubsampleSpec(
            name="full_grid",
            description="All views (baseline run).",
            selector=select_full,
        ),
        SubsampleSpec(
            name="center_cross",
            description="Central row and column.",
            selector=select_center_cross,
        ),
        SubsampleSpec(
            name="outer_ring",
            description="Only perimeter cameras.",
            selector=select_outer_ring,
        ),
        SubsampleSpec(
            name="even_grid",
            description="Every other row and column (stride 2).",
            selector=select_even_grid,
        ),
        SubsampleSpec(
            name="checkerboard",
            description="Half-resolution checkerboard pattern.",
            selector=select_checkerboard,
        ),
        SubsampleSpec(
            name=f"random_{random_count}",
            description=f"Random sample of {random_count} views (seeded).",
            selector=make_random_selector(random_count),
        ),
    ]


def write_subset_file(destination: Path, selection: Iterable[ImageView]) -> None:
    """Persist the selected image names for the trainer."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for view in selection:
            handle.write(f"{view.name}\n")


def build_trainer_command(
    python_cmd: str,
    data_dir: Path,
    result_dir: Path,
    train_list: Path,
    data_factor: int,
    extra_args: List[str] | None,
) -> List[str]:
    """Assemble the command line for a single training run."""
    command = [
        python_cmd,
        "examples/simple_trainer.py",
        "default",
        "--disable_viewer",
        "--data_factor",
        str(data_factor),
        "--data_dir",
        str(data_dir),
        "--result_dir",
        str(result_dir),
        "--save_ply",
        "--pose_opt",
        "--eval_steps",
        "3000",
        "7000",
        "30000",
        "--ply_steps",
        "3000",
        "7000",
        "30000",
        "--strategy.reset_every",
        "100000",
        "--strategy.pause_refine_after_reset",
        "0",
        "--strategy.prune_scale3d",
        "0.22",
        "--strategy.prune_scale2d",
        "0.12",
        "--strategy.prune_opa",
        "0.006",
        "--strategy.grow_grad2d",
        "0.00035",
        "--strategy.grow_scale3d",
        "0.012",
        "--strategy.refine_stop_iter",
        "26000",
        "--strategy.refine_scale2d_stop_iter",
        "26000",
        "--scale_reg",
        "0.0005",
        "--scales_lr",
        "0.003",
        "--means_lr",
        "0.00012",
        "--antialiased",
        "--train_list",
        str(train_list),
    ]
    if extra_args:
        command.extend(extra_args)
    return command


def format_command(command: Sequence[str]) -> str:
    """Return a shell-friendly rendering of a command list."""
    return " ".join(shlex.quote(token) for token in command)


def resolve_path(candidate: str, repo_root: Path, must_exist: bool = True) -> Path:
    """Normalise a user-supplied path against the repository root."""
    path = Path(candidate).expanduser()
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(f"Resolved path '{path}' does not exist.")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Launch multiple 3DGS runs on different light-field subsamples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        default="../gs7/input_data/dog/dog_lfr/inner_02",
        help="Path to the COLMAP dataset root (expects 'images/' and 'sparse/').",
    )
    parser.add_argument(
        "--result-root",
        default="results/dog_lfr_subsamples",
        help="Directory that will contain per-subsample result folders.",
    )
    parser.add_argument(
        "--data-factor",
        type=int,
        default=1,
        help="Dataset downsample factor passed to the trainer.",
    )
    parser.add_argument(
        "--random-count",
        type=int,
        default=25,
        help="Number of views to keep for the random subsample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for random subsampling.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        help="Optional subset of strategy names to run (others will be skipped).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs that already have a result directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to use for launching the trainer.",
    )
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        help="Additional arguments appended to the trainer command. "
        "Use '--' to separate script arguments from trainer arguments.",
    )

    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    if any(sep in args.python for sep in ("/", "\\")) or args.python.startswith("."):
        python_cmd = str(Path(args.python).expanduser())
    else:
        python_cmd = args.python

    dataset_root = resolve_path(args.data_dir, repo_root, must_exist=True)
    result_root = resolve_path(args.result_root, repo_root, must_exist=False)
    image_dir = dataset_root / "images"

    images, context = parse_image_grid(image_dir)
    specs = build_subsample_specs(args.random_count)
    if args.only:
        allowed = set(args.only)
        specs = [spec for spec in specs if spec.name in allowed]
        missing = allowed - {spec.name for spec in specs}
        if missing:
            raise ValueError(
                f"Requested subset names not recognised: {', '.join(sorted(missing))}"
            )

    if not specs:
        print("No subsampling strategies selected; exiting.", file=sys.stderr)
        return 1

    print(
        f"Found light-field grid with {context.row_count} rows, "
        f"{context.col_count} cols, and {len(images)} images."
    )

    for index, spec in enumerate(specs):
        rng = random.Random(args.seed + index)
        selection = spec.selector(images, context, rng)
        if not selection:
            print(
                f"[{spec.name}] Skipping because selector produced no images.",
                file=sys.stderr,
            )
            continue
        subset_dir = result_root / spec.name
        subset_file = subset_dir / "train_images.txt"
        ckpt_dir = subset_dir / "ckpts"
        has_ckpts = ckpt_dir.exists() and any(ckpt_dir.iterdir())

        command = build_trainer_command(
            python_cmd=python_cmd,
            data_dir=dataset_root,
            result_dir=subset_dir,
            train_list=subset_file,
            data_factor=args.data_factor,
            extra_args=args.extra_args,
        )

        try:
            display_path = subset_file.relative_to(repo_root)
        except ValueError:
            display_path = subset_file
        message_lines = [
            f"\n[{spec.name}] {spec.description}",
            f"  Views: {len(selection)} -> {display_path}",
            f"  Command: {format_command(command)}",
        ]
        if args.skip_existing and has_ckpts:
            message_lines.append(
                f"  (skipping — checkpoints already exist in {ckpt_dir})"
            )
        print("\n".join(message_lines))

        if args.skip_existing and has_ckpts:
            continue
        if args.dry_run:
            continue

        subset_dir.mkdir(parents=True, exist_ok=True)
        write_subset_file(subset_file, selection)

        subprocess.run(command, check=True, cwd=repo_root)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
