#!/usr/bin/env python3
"""
Eval-only runner for existing Difix3D dual-training runs.

Reads Difix3D SLURM .out logs to recover (result_dir, eval_dataset) pairs and
kicks off eval-only jobs using the Difix simple trainer. This is useful when
training completed but external eval failed to write stats.

Usage examples:

  # Dry-run: show the commands that would be executed
  python scripts/eval_difix_from_logs.py

  # Actually run eval for all discovered runs that are missing stats
  python scripts/eval_difix_from_logs.py --run

  # Restrict to specific scenes or variants
  python scripts/eval_difix_from_logs.py --run --scenes dog optics --variants combined filtered

Notes:
  - Defaults assume repo layout where this script lives in gsplat/, and Difix3D
    sits next to it: ../Difix3D.
  - For each run, this script picks the newest ckpt_*_rank0.pt.
  - It prefers covisible alignment if present, otherwise falls back to
    alignments/test_to_train.npz.
  - It passes --test_every 1 to keep a val-only eval (files tagged '_eval_').
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent
DIFIX_ROOT = (REPO_ROOT.parent / "Difix3D").resolve()


@dataclass
class EvalPlan:
    scene: str
    modality: str  # 'iphone' or 'stereo'
    variant: str   # 'combined' or 'filtered'
    result_dir: Path
    eval_dir: Path
    ckpt_path: Path
    out_dir: Path
    align_path: Optional[Path]
    covi_dir: Optional[Path]


RESULT_RE = re.compile(r"^Result dir:\s*(?P<path>\S.*?)\s*$")
EVAL_RE = re.compile(r"^Eval dataset:\s*(?P<path>\S.*?)\s*$")


def discover_from_logs(logs_root: Path) -> Dict[Path, Path]:
    """Return mapping: result_dir -> eval_dir by parsing SLURM .out logs."""
    mapping: Dict[Path, Path] = {}
    for entry in logs_root.glob("*.out"):
        result_dir: Optional[Path] = None
        eval_dir: Optional[Path] = None
        try:
            with entry.open("r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    m1 = RESULT_RE.search(line)
                    if m1:
                        result_dir = Path(m1.group("path")).resolve()
                    m2 = EVAL_RE.search(line)
                    if m2:
                        eval_dir = Path(m2.group("path")).resolve()
        except Exception:
            continue
        if result_dir and eval_dir:
            mapping[result_dir] = eval_dir
    return mapping


def choose_ckpt(ckpt_dir: Path) -> Optional[Path]:
    files = sorted(ckpt_dir.glob("ckpt_*_rank0.pt"))
    return files[-1] if files else None


def build_plan(result_dir: Path, eval_dir: Path) -> Optional[EvalPlan]:
    try:
        scene = result_dir.parent.parent.name
        modality = result_dir.parent.name
        variant = result_dir.name
    except Exception:
        return None

    ckpt = choose_ckpt(result_dir / "ckpts")
    if not ckpt:
        return None

    out_dir = result_dir / "eval_on_test"

    # Prefer covisible alignment if available
    covi_align = result_dir / "covisible" / "test" / "alignment.npz"
    plain_align = result_dir / "alignments" / "test_to_train.npz"
    align_path = covi_align if covi_align.exists() else (plain_align if plain_align.exists() else None)
    covi_dir = result_dir / "covisible" / "test" / "1x"
    if not covi_dir.exists():
        covi_dir = None

    return EvalPlan(
        scene=scene,
        modality=modality,
        variant=variant,
        result_dir=result_dir,
        eval_dir=eval_dir,
        ckpt_path=ckpt,
        out_dir=out_dir,
        align_path=align_path,
        covi_dir=covi_dir,
    )


def has_stats(plan: EvalPlan) -> bool:
    stats_dir = plan.out_dir / "stats"
    return any(stats_dir.glob("val_step*.json"))


def cmd_for(plan: EvalPlan) -> List[str]:
    trainer = DIFIX_ROOT / "examples" / "gsplat" / "simple_trainer_difix3d.py"
    cmd: List[str] = [
        "python",
        str(trainer),
        "default",
        "--disable_viewer",
        "--data_factor",
        "1",
        "--data_dir",
        str(plan.eval_dir),
        "--result_dir",
        str(plan.out_dir),
        "--ckpt",
        str(plan.ckpt_path),
        "--test_every",
        "1",
        "--max_steps",
        "30000",
        "--eval_only",
    ]
    if plan.align_path:
        cmd += ["--dataset-transform-path", str(plan.align_path)]
    if plan.covi_dir:
        cmd += ["--eval_use_covisible", "--eval_covisible_dir", str(plan.covi_dir)]
    return cmd


def run(cmd: List[str]) -> int:
    print("Executing:", shlex.join(cmd))
    proc = subprocess.run(cmd)
    return proc.returncode


def main() -> None:
    ap = argparse.ArgumentParser(description="Eval-only for Difix3D runs based on logs")
    ap.add_argument("--difix-root", type=Path, default=DIFIX_ROOT, help="Path to Difix3D repo root")
    ap.add_argument("--logs-root", type=Path, default=DIFIX_ROOT / "slurm_logs", help="Path to slurm logs")
    ap.add_argument("--results-root", type=Path, default=DIFIX_ROOT / "results", help="Root of Difix results")
    ap.add_argument("--scenes", nargs="*", default=None, help="Optional list of scenes to include")
    ap.add_argument("--variants", nargs="*", default=None, choices=["combined", "filtered"], help="Variants to include")
    ap.add_argument("--force", action="store_true", help="Run even if stats already exist")
    ap.add_argument("--run", action="store_true", help="Execute eval commands (default: dry-run)")
    args = ap.parse_args()

    mapping = discover_from_logs(args.logs_root)
    if not mapping:
        print("No (result_dir, eval_dir) pairs found in logs.")
        return

    plans: List[EvalPlan] = []
    for result_dir, eval_dir in mapping.items():
        if not result_dir.exists():
            continue
        plan = build_plan(result_dir, eval_dir)
        if not plan:
            continue
        if args.scenes and plan.scene not in set(args.scenes):
            continue
        if args.variants and plan.variant not in set(args.variants):
            continue
        plans.append(plan)

    if not plans:
        print("No eligible runs discovered.")
        return

    print(f"Discovered {len(plans)} runs.")
    for plan in plans:
        already = has_stats(plan)
        tag = f"{plan.scene}/{plan.modality}/{plan.variant}"
        cmd = cmd_for(plan)
        if already and not args.force:
            print(f"[skip] {tag}: stats already exist at {plan.out_dir}/stats")
            continue
        if args.run:
            rc = run(cmd)
            if rc != 0:
                print(f"[fail] {tag}: return code {rc}")
        else:
            print(shlex.join(cmd))


if __name__ == "__main__":
    main()
