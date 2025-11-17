#!/usr/bin/env python3
"""
Generate LaTeX tables with scenes as columns and grouped methods as rows.
Creates one combined table with PSNR, SSIM, and LPIPS metrics, organized into
two sections: iPhone Capture and Stereo Capture.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


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

BASELINE_SOURCE = "baseline"
DIFIX_SOURCE = "difix"


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    modality: str
    variant: str
    source: str


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIFIX_RESULTS_ROOT = (REPO_ROOT.parent / "Difix3D" / "results").as_posix()

# Grouped layout with improved headings and labels.
GROUPS: Tuple[Tuple[str, Tuple[MethodSpec, ...]], ...] = (
    (
        "iPhone Capture",
        (
            MethodSpec("iphone_filtered", "Monocular", "iphone", "filtered", BASELINE_SOURCE),
            MethodSpec(
                "iphone_filtered_difix",
                r"Monocular w/ Prior",
                "iphone",
                "filtered",
                DIFIX_SOURCE,
            ),
            MethodSpec("iphone_combined", "iPhone", "iphone", "combined", BASELINE_SOURCE),
            MethodSpec(
                "iphone_combined_difix",
                r"iPhone w/ Prior",
                "iphone",
                "combined",
                DIFIX_SOURCE,
            ),
        ),
    ),
    (
        "Stereo Capture",
        (
            MethodSpec("stereo_filtered", "Monocular", "stereo", "filtered", BASELINE_SOURCE),
            MethodSpec(
                "stereo_filtered_difix",
                r"Monocular w/ Prior",
                "stereo",
                "filtered",
                DIFIX_SOURCE,
            ),
            MethodSpec("stereo_combined", "Stereo", "stereo", "combined", BASELINE_SOURCE),
            MethodSpec(
                "stereo_combined_difix",
                r"Stereo w/ Prior",
                "stereo",
                "combined",
                DIFIX_SOURCE,
            ),
        ),
    ),
)
METRICS: Tuple[str, ...] = ("psnr", "ssim", "lpips")

STEP_PATTERN = re.compile(r"val_step(\d+)\.json$")

# Metric formatting
METRIC_FORMATS = {
    "psnr": "{:.2f}",
    "ssim": "{:.3f}",
    "lpips": "{:.3f}",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate formatted LaTeX tables with scenes as columns."
    )
    parser.add_argument(
        "--results-root",
        default="results",
        help="Path to the results directory (default: %(default)s).",
    )
    parser.add_argument(
        "--difix-results-root",
        default=DEFAULT_DIFIX_RESULTS_ROOT,
        help="Path to the Difix results directory (default: %(default)s).",
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

    if not {"psnr", "ssim", "lpips"} <= payload.keys():
        return None
    return {metric: float(payload[metric]) for metric in ("psnr", "ssim", "lpips")}


def load_method_metrics(
    root: Path,
    scene: str,
    spec: MethodSpec,
) -> Optional[Dict[str, float]]:
    """Load metrics for a specific method from eval_on_test stats."""
    stats_dir = (
        root
        / scene
        / spec.modality
        / spec.variant
        / "eval_on_test"
        / "stats"
    )
    return load_latest_metrics(stats_dir)


def collect_all_metrics(
    results_roots: Dict[str, Optional[Path]],
    method_specs: Tuple[MethodSpec, ...],
) -> Dict[str, Dict[str, Optional[Dict[str, float]]]]:
    """
    Collect metrics for all requested methods.
    Returns: {scene: {method_key: metrics or None}}
    """
    data: Dict[str, Dict[str, Optional[Dict[str, float]]]] = {}

    for scene in SCENES:
        data[scene] = {}
        for spec in method_specs:
            root = results_roots.get(spec.source)
            metrics: Optional[Dict[str, float]] = None
            if root is not None:
                metrics = load_method_metrics(root, scene, spec)
            if metrics is None:
                print(
                    f"[WARN] Missing metrics for {scene}/{spec.modality}/{spec.variant}"
                    f" ({spec.label})",
                    file=sys.stderr,
                )
            data[scene][spec.key] = metrics

    return data


def format_scene_name(scene: str) -> str:
    """Format scene name for display (capitalize words, remove hyphens)."""
    parts = scene.split("-")
    return " ".join(word.capitalize() for word in parts)


def build_metric_tabular(
    data: Dict[str, Dict[str, Optional[Dict[str, float]]]],
    metric: str,
    groups: Tuple[Tuple[str, Tuple[MethodSpec, ...]], ...],
) -> str:
    """Build a single tabular environment for the specified metric."""
    fmt = METRIC_FORMATS[metric]

    # Build header
    scene_headers = [format_scene_name(scene) for scene in SCENES]
    header = " & ".join(["Method"] + scene_headers + ["Avg", "Std"]) + r" \\"

    # Build rows grouped by capture type
    rows: List[str] = []
    num_cols = len(SCENES) + 3  # scenes + method + avg + std
    for group_title, specs in groups:
        rows.append(rf"\multicolumn{{{num_cols}}}{{l}}{{\textbf{{{group_title}}}}} \\")
        rows.append(r"\addlinespace[2pt]")
        for spec in specs:
            cells = [spec.label]
            values: List[float] = []

            for scene in SCENES:
                metrics = data[scene][spec.key]
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
        rows.append(r"\addlinespace[4pt]")

    # Build tabular
    num_cols = len(SCENES) + 3  # scenes + method + avg + std
    col_spec = "l" + "c" * (num_cols - 1)

    arrow = r"$\uparrow$" if metric in ("psnr", "ssim") else r"$\downarrow$"
    metric_upper = metric.upper()

    tabular = (
        f"\\textbf{{{metric_upper}{arrow}}}\n"
        r"{\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.05}%" "\n"
        r"\begin{tabular}{" + col_spec + "}\n"
        r"\toprule" "\n"
        f"{header}\n"
        r"\midrule" "\n"
        + "\n".join(rows) + "\n"
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
        r"}"
    )

    return tabular


def build_combined_per_scene_table(
    data: Dict[str, Dict[str, Optional[Dict[str, float]]]],
    groups: Tuple[Tuple[str, Tuple[MethodSpec, ...]], ...],
) -> str:
    """Build a combined table with all three metrics."""
    tabulars = []
    for metric in METRICS:
        tabular = build_metric_tabular(data, metric, groups)
        tabulars.append(tabular)

    table = (
        r"\begin{table*}[ht]" "\n"
        r"\centering" "\n"
        r"\small" "\n\n"
        + "\n\n\\vspace{0.75em}\n\n".join(tabulars) + "\n\n"
        r"\caption{Per-scene metrics for dual-training evaluation grouped by capture modality.}" "\n"
        r"\label{tab:dual-per-scene}" "\n"
        r"\end{table*}"
    )

    return table


def build_summary_table(
    data: Dict[str, Dict[str, Optional[Dict[str, float]]]],
    groups: Tuple[Tuple[str, Tuple[MethodSpec, ...]], ...],
) -> str:
    """Build summary table with average metrics across all scenes."""
    rows: List[str] = []

    for group_title, specs in groups:
        rows.append(rf"\multicolumn{{4}}{{l}}{{\textbf{{{group_title}}}}} \\")
        for spec in specs:
            method_name = spec.label

            # Collect all metrics across scenes
            metric_values: Dict[str, List[float]] = {m: [] for m in METRICS}

            for scene in SCENES:
                metrics = data[scene][spec.key]
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
        rows.append(r"\addlinespace[4pt]")

    # Build table
    table = (
        r"\begin{table}[ht]" "\n"
        r"\centering" "\n"
        r"\small" "\n"
        r"\begin{tabular}{lrrr}" "\n"
        r"\toprule" "\n"
        r"Method & PSNR$\uparrow$ & SSIM$\uparrow$ & LPIPS$\downarrow$ \\" "\n"
        r"\midrule" "\n"
        + "\n".join(rows) + "\n"
        r"\bottomrule" "\n"
        r"\end{tabular}%" "\n"
        r"\caption{Average metrics across all scenes for dual-training evaluation grouped by capture modality.}" "\n"
        r"\label{tab:dual-summary}" "\n"
        r"\end{table}"
    )

    return table


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    difix_root = Path(args.difix_results_root)

    results_roots: Dict[str, Optional[Path]] = {
        BASELINE_SOURCE: results_root,
        DIFIX_SOURCE: difix_root,
    }

    # Collect all metrics for the requested grouped rows
    flat_specs: Tuple[MethodSpec, ...] = tuple(spec for _, specs in GROUPS for spec in specs)
    data = collect_all_metrics(results_roots, flat_specs)

    # Build combined per-scene table
    per_scene_table = build_combined_per_scene_table(data, GROUPS)

    # Build summary table
    summary_table = build_summary_table(data, GROUPS)

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
