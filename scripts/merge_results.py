#!/usr/bin/env python3
"""Merge phase3 result folders into a single results/ directory."""
from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy import stats as sp_stats
except ImportError:
    sp_stats = None

PHASE3 = Path(__file__).resolve().parent
OUT = PHASE3 / "results"
ARCHIVE = PHASE3 / "_results_sessions_archive"

# Folders to merge (order: later folders override on same mtime tie — newest mtime wins)
SOURCE_DIRS = [
    PHASE3 / "results",
    PHASE3 / "results 4",
    PHASE3 / "results copy",
    PHASE3 / "results_drive_snapshot",
    PHASE3 / "results ablation partial",
    PHASE3 / "results 6",
    PHASE3 / "results 7",
    PHASE3 / "results 8",
    PHASE3 / "results 9",
    PHASE3 / "results 10",
]

MAIN_MODELS = ["standard", "de_full", "de_best", "jde", "diff"]
ABLATION_MODELS = [
    "standard", "diff", "de_full", "de_best", "jde",
    "de_avg3", "de_learned", "de_sum", "de_nobase", "de_indepV",
]


def collect_json_sources() -> dict[str, tuple[Path, float]]:
    """Map filename -> (source_path, mtime); newest wins."""
    chosen: dict[str, tuple[Path, float]] = {}
    for src_dir in SOURCE_DIRS:
        if not src_dir.is_dir():
            continue
        for p in src_dir.glob("*.json"):
            mt = p.stat().st_mtime
            prev = chosen.get(p.name)
            if prev is None or mt >= prev[1]:
                chosen[p.name] = (p, mt)
    return chosen


def merge_into_results(chosen: dict[str, tuple[Path, float]]) -> list[tuple[str, str]]:
    OUT.mkdir(parents=True, exist_ok=True)
    log: list[tuple[str, str]] = []
    for name, (src, _) in sorted(chosen.items()):
        dst = OUT / name
        if dst.exists() and dst.read_bytes() == src.read_bytes():
            log.append((name, "unchanged"))
        else:
            shutil.copy2(src, dst)
            log.append((name, f"from {src.parent.name}"))
    return log


def load_runs() -> list[dict]:
    rows = []
    for p in sorted(OUT.glob("*.json")):
        if "_msweep" in p.name:
            continue
        rows.append(json.loads(p.read_text()))
    return rows


def load_msweep() -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(OUT.glob("*_msweep.json"))]


def rows_to_summary(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    metrics = [c for c in df.columns if c.startswith("test_")]
    agg = df.groupby(["dataset", "model"])[metrics].agg(["mean", "std"]).reset_index()
    agg.columns = [
        "_".join(c).strip("_") if isinstance(c, tuple) else c for c in agg.columns
    ]
    return agg


def paired_ttests(rows: list[dict]) -> pd.DataFrame:
    records = []
    by_ds = defaultdict(list)
    for r in rows:
        by_ds[r["dataset"]].append(r)
    for ds, ds_rows in by_ds.items():
        for baseline, label in [("standard", "DE vs Standard"), ("diff", "DE vs Diff")]:
            de_vals, base_vals = [], []
            seeds = sorted({r["seed"] for r in ds_rows if r["model"] == "de_full"})
            for seed in seeds:
                de = next((r for r in ds_rows if r["model"] == "de_full" and r["seed"] == seed), None)
                ba = next((r for r in ds_rows if r["model"] == baseline and r["seed"] == seed), None)
                if de and ba:
                    de_vals.append(de["test_acc"])
                    base_vals.append(ba["test_acc"])
            if len(de_vals) >= 2 and sp_stats is not None:
                diff = np.array(de_vals) - np.array(base_vals)
                _, p = sp_stats.ttest_rel(de_vals, base_vals)
                ci_low, ci_high = sp_stats.t.interval(
                    0.95, len(diff) - 1, loc=diff.mean(), scale=sp_stats.sem(diff)
                )
                records.append({
                    "dataset": ds,
                    "comparison": label,
                    "p_value": float(p),
                    "mean_diff": float(diff.mean()),
                    "ci_low": float(ci_low),
                    "ci_high": float(ci_high),
                    "n_seeds": len(de_vals),
                })
    return pd.DataFrame(records)


def completeness_report() -> str:
    runs: dict[tuple[str, str], set[int]] = defaultdict(set)
    msweep: dict[str, set[int]] = defaultdict(set)
    for p in OUT.glob("*.json"):
        name = p.stem
        if "_msweep" in name:
            parts = name.replace("_msweep", "").split("_")
            ds, seed_s = parts[0], parts[-1].replace("seed", "")
            msweep[ds].add(int(seed_s))
            continue
        parts = name.split("_")
        seed = int(parts[-1].replace("seed", ""))
        model = "_".join(parts[1:-1])
        ds = parts[0]
        runs[(ds, model)].add(seed)

    lines = ["# Phase 3 merged results — completeness\n"]
    for ds in sorted({k[0] for k in runs}):
        lines.append(f"\n## {ds.upper()}\n")
        lines.append("| model | seeds | count |")
        lines.append("|-------|-------|-------|")
        for (d, model), seeds in sorted(runs.items()):
            if d != ds:
                continue
            s = sorted(seeds)
            lines.append(f"| {model} | {s} | {len(s)} |")
        if ds in msweep:
            s = sorted(msweep[ds])
            lines.append(f"\n**m_sweep (de_full):** seeds {s} ({len(s)} runs)\n")

    lines.append("\n## Missing (recommended next Colab runs)\n")
    for ds in ["imdb", "twitter", "yelp"]:
        for model in MAIN_MODELS:
            have = runs.get((ds, model), set())
            missing = [i for i in range(10) if i not in have]
            if missing and have:
                lines.append(f"- `{ds}` / `{model}`: missing seeds {missing}")
        for model in ABLATION_MODELS:
            if model in MAIN_MODELS:
                continue
            have = runs.get((ds, model), set())
            missing = [i for i in range(10) if i not in have]
            if missing and have:
                lines.append(f"- `{ds}` / `{model}` (ablation): missing seeds {missing}")
            elif not have and ds in {k[0] for k in runs}:
                lines.append(f"- `{ds}` / `{model}` (ablation): **no runs yet**")
    return "\n".join(lines) + "\n"


def archive_session_dirs():
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    for src_dir in SOURCE_DIRS:
        if not src_dir.is_dir() or src_dir.resolve() == OUT.resolve():
            continue
        dst = ARCHIVE / src_dir.name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(str(src_dir), str(dst))


def main() -> None:
    chosen = collect_json_sources()
    print(f"Merging {len(chosen)} unique JSON files into {OUT}")
    log = merge_into_results(chosen)
    from_sources = defaultdict(int)
    for _, action in log:
        if action.startswith("from "):
            from_sources[action.replace("from ", "")] += 1
    print("Copied from:", dict(from_sources))

    rows = load_runs()
    main_rows = [r for r in rows if r.get("model") in MAIN_MODELS]
    abl_rows = [r for r in rows if r.get("model") in ABLATION_MODELS]

    if main_rows:
        rows_to_summary(main_rows).to_csv(OUT / "summary_main.csv", index=False)
        print(f"Wrote summary_main.csv ({len(main_rows)} runs)")
    if abl_rows:
        rows_to_summary(abl_rows).to_csv(OUT / "summary_ablation.csv", index=False)
        print(f"Wrote summary_ablation.csv ({len(abl_rows)} runs)")

    ms = load_msweep()
    if ms:
        pd.DataFrame(ms).to_csv(OUT / "summary_m_sweep.csv", index=False)
        print(f"Wrote summary_m_sweep.csv ({len(ms)} runs)")

    stats_df = paired_ttests(main_rows)
    if not stats_df.empty:
        stats_df.to_csv(OUT / "stats_paired_ttest.csv", index=False)
        print("Wrote stats_paired_ttest.csv")

    manifest = completeness_report()
    (OUT / "MERGE_MANIFEST.md").write_text(manifest, encoding="utf-8")
    print(f"Wrote {OUT / 'MERGE_MANIFEST.md'}")

    archive_session_dirs()
    print(f"Moved session folders to {ARCHIVE}")


if __name__ == "__main__":
    main()
