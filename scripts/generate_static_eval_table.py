#!/usr/bin/env python3
"""
Generate LaTeX tables for static training with scenes as columns and camera types as rows.
Creates one combined table with PSNR, SSIM, and LPIPS metrics.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import statistics


SCENES: Tuple[str, ...] = (
    "action-figure",
    "ball",
    "chicken",
    "dog",
    "espresso",
    "optics",
    "salt-pepper",
    "shelf",
)
CAMERAS: Tuple[str, ...] = ("monocular", "iphone", "stereo", "lightfield")
METRICS: Tuple[str, ...] = ("mpsnr", "mssim", "mlpips")
METRIC_FALLBACK: Dict[str, str] = {
    "mpsnr": "psnr",
    "mssim": "ssim",
    "mlpips": "lpips",
}

STEP_PATTERN = re.compile(r"val_step(\d+)\.json$")

# Metric formatting
METRIC_FORMATS = {
    "mpsnr": "{:.2f}",
    "mssim": "{:.3f}",
    "mlpips": "{:.3f}",
}
METRIC_LABELS = {
    "mpsnr": "mPSNR",
    "mssim": "mSSIM",
    "mlpips": "mLPIPS",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate formatted LaTeX tables for static training with scenes as columns."
    )
    parser.add_argument(
        "--results-root",
        default="results/static",
        help="Path to the static results directory (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .tex file for per-scene tables.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Output .tex file for summary table.",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        default=None,
        help="Comma-separated list of cameras (defaults to iphone,lightfield,monocular,stereo).",
    )
    return parser.parse_args()


def load_latest_metrics(stats_dir: Path) -> Optional[Dict[str, float]]:
    """Load metrics from the latest validation step JSON."""
    if not stats_dir.is_dir():
        return None

    best_path: Optional[Path] = None
    best_step = -1

    for candidate in stats_dir.glob("val_step*.json"):
        match = STEP_PATTERN.match(candidate.name)
        if not match:
            continue
        step = int(match.group(1))
        if step > best_step:
            best_step = step
            best_path = candidate

    if best_path is None:
        return None

    with best_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    results: Dict[str, float] = {}
    for metric in METRICS:
        if metric in payload:
            results[metric] = float(payload[metric])
        else:
            fallback = METRIC_FALLBACK.get(metric)
            if fallback and fallback in payload:
                results[metric] = float(payload[fallback])
            else:
                return None
    return results


def collect_all_metrics(
    results_root: Path,
    cameras: List[str],
) -> Dict[str, Dict[str, Optional[Dict[str, float]]]]:
    """
    Collect metrics for all combinations of scene/camera.
    Returns: {scene: {camera: metrics or None}}
    """
    data: Dict[str, Dict[str, Optional[Dict[str, float]]]] = {}

    for scene in SCENES:
        data[scene] = {}
        for camera in cameras:
            stats_dir = (
                results_root
                / scene
                / camera
                / "eval_on_iphone_eval"
                / "stats"
            )
            metrics = load_latest_metrics(stats_dir)
            if metrics is None:
                print(
                    f"[WARN] Missing metrics for {scene}/{camera}",
                    file=sys.stderr,
                )
            data[scene][camera] = metrics

    return data


def format_scene_name(scene: str) -> str:
    """Format scene name for display (capitalize words, remove hyphens)."""
    parts = scene.split("-")
    return " ".join(word.capitalize() for word in parts)


def format_camera_name(camera: str) -> str:
    """Format camera name for display."""
    if camera == "iphone":
        return "iPhone"
    return camera.capitalize()


def build_metric_tabular(
    data: Dict[str, Dict[str, Optional[Dict[str, float]]]],
    metric: str,
    cameras: List[str],
) -> str:
    """Build a single tabular environment for the specified metric."""
    fmt = METRIC_FORMATS[metric]

    # Build header
    scene_headers = [format_scene_name(scene) for scene in SCENES]
    header = " & ".join(["Method"] + scene_headers + ["Avg", "Std"]) + r" \\"

    # Build rows for each camera type
    rows: List[str] = []
    for camera in cameras:
        method_name = format_camera_name(camera)
        cells = [method_name]
        values: List[float] = []

        for scene in SCENES:
            metrics = data[scene][camera]
            if metrics is not None and metric in metrics:
                value = metrics[metric]
                cells.append(fmt.format(value))
                values.append(value)
            else:
                cells.append("--")

        # Compute average and std
        if values:
            avg = statistics.mean(values)
            std = statistics.pstdev(values)
            cells.append(fmt.format(avg))
            cells.append(fmt.format(std))
        else:
            cells.append("--")
            cells.append("--")

        rows.append(" & ".join(cells) + r" \\")

    # Build tabular
    num_cols = len(SCENES) + 3  # scenes + method + avg + std
    col_spec = "l" + "c" * (num_cols - 1)

    arrow = r"$\uparrow$" if metric in ("mpsnr", "mssim") else r"$\downarrow$"
    metric_label = METRIC_LABELS.get(metric, metric.upper())

    tabular = (
        f"\\textbf{{{metric_label}{arrow}}}\n"
        r"\begin{tabular}{" + col_spec + "}\n"
        r"\hline" "\n"
        f"{header}\n"
        r"\hline" "\n"
        + "\n".join(rows) + "\n"
        r"\hline" "\n"
        r"\end{tabular}"
    )

    return tabular


def build_combined_per_scene_table(
    data: Dict[str, Dict[str, Optional[Dict[str, float]]]],
    cameras: List[str],
) -> str:
    """Build a combined table with all three metrics."""
    tabulars = []
    for metric in METRICS:
        tabular = build_metric_tabular(data, metric, cameras)
        tabulars.append(tabular)

    table = (
        r"\begin{table*}[ht]" "\n"
        r"\centering" "\n\n"
        + "\n\n\\vspace{1em}\n\n".join(tabulars) + "\n\n"
        r"\caption{Per-scene metrics for static training evaluation.}" "\n"
        r"\label{tab:static-per-scene}" "\n"
        r"\end{table*}"
    )

    return table


def build_summary_table(
    data: Dict[str, Dict[str, Optional[Dict[str, float]]]],
    cameras: List[str],
) -> str:
    """Build summary table with average metrics across all scenes."""
    rows: List[str] = []

    for camera in cameras:
        method_name = format_camera_name(camera)

        # Collect all metrics across scenes
        metric_values: Dict[str, List[float]] = {m: [] for m in METRICS}

        for scene in SCENES:
            metrics = data[scene][camera]
            if metrics is not None:
                for metric in METRICS:
                    if metric in metrics:
                        metric_values[metric].append(metrics[metric])

        # Compute averages
        cells = [method_name]
        for metric in METRICS:
            values = metric_values[metric]
            if values:
                avg = statistics.mean(values)
                fmt = METRIC_FORMATS[metric]
                cells.append(fmt.format(avg))
            else:
                cells.append("--")

        rows.append(" & ".join(cells) + r" \\")

    # Build table
    table = (
        r"\begin{table}[ht]" "\n"
        r"\centering" "\n"
        r"\begin{tabular}{lrrr}" "\n"
        r"\hline" "\n"
        r"Method & mPSNR$\uparrow$ & mSSIM$\uparrow$ & mLPIPS$\downarrow$ \\" "\n"
        r"\hline" "\n"
        + "\n".join(rows) + "\n"
        r"\hline" "\n"
        r"\end{tabular}%" "\n"
        r"\caption{Average metrics across all scenes for static training evaluation.}" "\n"
        r"\label{tab:static-summary}" "\n"
        r"\end{table}"
    )

    return table


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)

    # Parse cameras
    if args.cameras:
        cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]
    else:
        cameras = list(CAMERAS)

    # Collect all metrics
    data = collect_all_metrics(results_root, cameras)

    # Build combined per-scene table
    per_scene_table = build_combined_per_scene_table(data, cameras)

    # Build summary table
    summary_table = build_summary_table(data, cameras)

    # Write outputs
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(per_scene_table + "\n", encoding="utf-8")
        print(f"Per-scene table written to {args.output}")
    else:
        print(per_scene_table)

    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(summary_table + "\n", encoding="utf-8")
        print(f"Summary table written to {args.summary_output}")
    else:
        print("\n\n" + summary_table)


if __name__ == "__main__":
    main()
