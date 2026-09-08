#!/usr/bin/env python3
"""Build Phase 3 LaTeX assets: polished tables, Twitter-extended A5, m-sweep figure."""
from __future__ import annotations

import ast
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent / "results"
LATEX = RESULTS / "latex"
LATEX.mkdir(parents=True, exist_ok=True)

MODEL_LABELS = {
    "standard": "Standard",
    "diff": "DiffTransformer",
    "de_full": "DE full",
    "de_best": "DE/best/1/bin",
    "jde": "jDE",
    "de_avg3": "Avg-3",
    "de_learned": "Learned",
    "de_sum": "Sum",
    "de_nobase": "No-base",
    "de_indepV": r"Indep-$V$",
}
ABLATION_ORDER = [
    "standard", "diff", "de_full", "de_best", "jde",
    "de_avg3", "de_learned", "de_sum", "de_nobase", "de_indepV",
]
MAIN_ORDER = ["standard", "de_full", "de_best", "jde", "diff"]


def fmt_pm(mean: float, std: float, scale: float = 100.0, digits: int = 2) -> str:
    return f"{mean * scale:.{digits}f} $\\pm$ {std * scale:.{digits}f}"


def student_t_ppf_approx(df: int, p: float = 0.975) -> float:
    """Approximate two-sided 95% t critical value (df>=1)."""
    # Rational approximation sufficient for reporting
    z = 1.959963984540054
    if df <= 0:
        return z
    g1 = (z**3 + z) / 4.0
    g2 = (5 * z**5 + 16 * z**3 + 3 * z) / 96.0
    g3 = (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / 384.0
    return z + g1 / df + g2 / df**2 + g3 / df**3


def paired_ttest(a: np.ndarray, b: np.ndarray):
    diff = a - b
    n = len(diff)
    mean = float(diff.mean())
    se = float(diff.std(ddof=1) / math.sqrt(n))
    if se == 0:
        return 1.0, mean, mean, mean, n
    t = mean / se
    # two-sided p via regularized incomplete beta approximation
    df = n - 1
    x = df / (df + t * t)
    # Incomplete beta Ix(a,b) with a=df/2, b=1/2 — use scipy-free continued fraction light approx
    # Fallback: use normal for large df else rough
    try:
        from scipy import stats as sp_stats
        _, p = sp_stats.ttest_rel(a, b)
        ci_low, ci_high = sp_stats.t.interval(0.95, df, loc=mean, scale=se)
        return float(p), mean, float(ci_low), float(ci_high), n
    except ImportError:
        # Normal approx for p when |t| large; better than nothing
        # Use complementary error function for two-sided normal p
        p = math.erfc(abs(t) / math.sqrt(2))
        # small-df correction: inflate p slightly
        if df < 30:
            p = min(1.0, p * (1 + 2.0 / df))
        tcrit = student_t_ppf_approx(df)
        return float(p), mean, mean - tcrit * se, mean + tcrit * se, n


def load_runs():
    rows = []
    for p in sorted(RESULTS.glob("*.json")):
        if "_msweep" in p.name:
            continue
        rows.append(json.loads(p.read_text()))
    return rows


def write_table3(df: pd.DataFrame) -> None:
    lines = [
        "% Auto-generated — paste into Manuscript_body.tex as tab:compact_results",
        "\\begin{table}[H]",
        "  \\centering",
        "  \\small",
        r"  \caption{\Rone[1.1, 2.8, 3.12]{Multi-seed test results (mean $\pm$ std) under the official-split protocol with validation early stopping. IMDb: $n{=}5$ seeds; Twitter: $n{=}10$ seeds. Macro F1 is reported alongside accuracy. Yelp is deferred (not re-run under this protocol). Best Acc per dataset in bold.}}",
        "  \\label{tab:compact_results}",
        "  \\begin{tabular}{llcc}",
        "    \\toprule",
        r"    \textbf{Dataset} & \textbf{Model} & \textbf{Acc (\%)} & \textbf{F1 macro (\%)} \\",
        "    \\midrule",
    ]
    for dataset in ["imdb", "twitter"]:
        ds_rows = df[df["dataset"] == dataset]
        # bold best mean acc
        best = None
        if not ds_rows.empty:
            best = ds_rows.loc[ds_rows["test_acc_mean"].idxmax(), "model"]
        first = True
        n_models = sum(1 for m in MAIN_ORDER if not ds_rows[ds_rows["model"] == m].empty)
        for model in MAIN_ORDER:
            row = ds_rows[ds_rows["model"] == model]
            if row.empty:
                continue
            r = row.iloc[0]
            acc = fmt_pm(r["test_acc_mean"], r["test_acc_std"])
            f1 = fmt_pm(r["test_f1_macro_mean"], r["test_f1_macro_std"])
            if model == best:
                acc = f"\\textbf{{{acc}}}"
            ds_cell = (
                f"\\multirow{{{n_models}}}{{*}}{{{dataset.upper()}}}" if first else ""
            )
            first = False
            lines.append(
                f"    {ds_cell} & {MODEL_LABELS.get(model, model)} & {acc} & {f1} \\\\"
            )
        lines.append("    \\midrule")
    if lines[-1].endswith("\\midrule"):
        lines[-1] = "    \\bottomrule"
    lines += ["  \\end{tabular}", "\\end{table}", ""]
    (LATEX / "table3_main.tex").write_text("\n".join(lines), encoding="utf-8")


def write_ablation(df: pd.DataFrame) -> None:
    lines = [
        "% Auto-generated ablation Table A1",
        "\\begin{table}[H]",
        "  \\centering",
        "  \\small",
        r"  \caption{\Rone[1.2, 3.13, 4.6]{\Rthree[3.1]{Parameter-matched ablation (mean $\pm$ std test accuracy, \%). "
        r"Avg-3 / Learned / Sum isolate combination structure; No-base removes $S_1$; Indep-$V$ uses a separate value projection. "
        r"IMDb: $n{=}5$; Twitter: $n{=}10$.}}}",
        "  \\label{tab:ablation_a1}",
        "  \\begin{tabular}{llc}",
        "    \\toprule",
        r"    \textbf{Dataset} & \textbf{Variant} & \textbf{Acc (\%)} \\",
        "    \\midrule",
    ]
    for dataset in ["imdb", "twitter"]:
        ds = df[df["dataset"] == dataset]
        first = True
        models = [m for m in ABLATION_ORDER if not ds[ds["model"] == m].empty]
        for model in models:
            r = ds[ds["model"] == model].iloc[0]
            ds_cell = f"\\multirow{{{len(models)}}}{{*}}{{{dataset.upper()}}}" if first else ""
            first = False
            lines.append(
                f"    {ds_cell} & {MODEL_LABELS.get(model, model)} "
                f"& {fmt_pm(r['test_acc_mean'], r['test_acc_std'])} \\\\"
            )
        lines.append("    \\midrule")
    if lines[-1].endswith("\\midrule"):
        lines[-1] = "    \\bottomrule"
    lines += ["  \\end{tabular}", "\\end{table}", ""]
    (LATEX / "table_a1_ablation.tex").write_text("\n".join(lines), encoding="utf-8")


def write_stats(rows: list[dict]) -> None:
    records = []
    by_ds = defaultdict(list)
    for r in rows:
        by_ds[r["dataset"]].append(r)
    for ds, ds_rows in sorted(by_ds.items()):
        for baseline, label in [("standard", "DE vs Standard"), ("diff", "DE vs Diff")]:
            de_vals, base_vals = [], []
            seeds = sorted({r["seed"] for r in ds_rows if r["model"] == "de_full"})
            for seed in seeds:
                de = next((r for r in ds_rows if r["model"] == "de_full" and r["seed"] == seed), None)
                ba = next((r for r in ds_rows if r["model"] == baseline and r["seed"] == seed), None)
                if de and ba:
                    de_vals.append(de["test_acc"])
                    base_vals.append(ba["test_acc"])
            if len(de_vals) >= 2:
                p, mean, lo, hi, n = paired_ttest(np.array(de_vals), np.array(base_vals))
                records.append({
                    "dataset": ds,
                    "comparison": label,
                    "p_value": p,
                    "mean_diff": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "n_seeds": n,
                })
    pdf = pd.DataFrame(records)
    pdf.to_csv(RESULTS / "stats_paired_ttest.csv", index=False)

    lines = [
        "% Auto-generated stats Table A5",
        "\\begin{table}[H]",
        "  \\centering",
        "  \\small",
        r"  \caption{\Rone[1.1, 3.12]{Paired $t$-tests on matched seeds: DEAttention (DE full) minus baseline (test accuracy). "
        r"Negative mean differences favour the baseline. IMDb $n{=}5$; Twitter $n{=}10$.}}",
        "  \\label{tab:stats_a5}",
        "  \\begin{tabular}{llccc}",
        "    \\toprule",
        r"    \textbf{Dataset} & \textbf{Comparison} & \textbf{$p$} & \textbf{Mean diff (pp)} & \textbf{95\% CI (pp)} \\",
        "    \\midrule",
    ]
    for _, r in pdf.iterrows():
        ci = f"[{r['ci_low']*100:.2f}, {r['ci_high']*100:.2f}]"
        lines.append(
            f"    {r['dataset'].upper()} & {r['comparison']} "
            f"& {r['p_value']:.4f} & {r['mean_diff']*100:.2f} & {ci} \\\\"
        )
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""]
    (LATEX / "table_a5_stats.tex").write_text("\n".join(lines), encoding="utf-8")


def parse_m_sweep_val(cell):
    if isinstance(cell, list):
        return cell
    return ast.literal_eval(cell)


def write_msweep_table_and_fig():
    df = pd.read_csv(RESULTS / "summary_m_sweep.csv")
    # Aggregate val Acc vs m across seeds
    records = []
    for _, row in df.iterrows():
        for item in parse_m_sweep_val(row["m_sweep_val"]):
            records.append({
                "dataset": row["dataset"],
                "seed": row["seed"],
                "m": item["m"],
                "val_acc": item["val_acc"],
            })
    long = pd.DataFrame(records)
    long.to_csv(LATEX / "fig_a1_m_sweep.csv", index=False)

    # Tabular mean±std val Acc by m
    lines = [
        "% Auto-generated m-sweep table",
        "\\begin{table}[H]",
        "  \\centering",
        "  \\small",
        r"  \caption{\Rone[1.6, 2.7]{Validation accuracy (mean $\pm$ std, \%) vs.\ mutation factor $m$ "
        r"for DE full ($n{=}10$ seeds per dataset). The seed-wise argmax of validation Acc is not fixed at $m{=}0.5$.}}",
        "  \\label{tab:detailed_results}",
        "  \\begin{tabular}{lccccc}",
        "    \\toprule",
        r"    \textbf{Dataset} & $m{=}0.1$ & $m{=}0.3$ & $m{=}0.5$ & $m{=}0.7$ & $m{=}0.9$ \\",
        "    \\midrule",
    ]
    for dataset in ["imdb", "twitter"]:
        cells = [dataset.upper()]
        for m in [0.1, 0.3, 0.5, 0.7, 0.9]:
            sub = long[(long["dataset"] == dataset) & (long["m"] == m)]["val_acc"]
            cells.append(fmt_pm(sub.mean(), sub.std(ddof=1) if len(sub) > 1 else 0.0))
        lines.append("    " + " & ".join(cells) + " \\\\")
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""]
    (LATEX / "table_a6_msweep.tex").write_text("\n".join(lines), encoding="utf-8")

    # Figure
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing — skipping PNG")
        return

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), sharey=False)
    for ax, dataset, title in zip(axes, ["imdb", "twitter"], ["IMDb", "Twitter"]):
        xs, means, stds = [], [], []
        for m in [0.1, 0.3, 0.5, 0.7, 0.9]:
            sub = long[(long["dataset"] == dataset) & (long["m"] == m)]["val_acc"]
            xs.append(m)
            means.append(100 * sub.mean())
            stds.append(100 * (sub.std(ddof=1) if len(sub) > 1 else 0.0))
        ax.errorbar(xs, means, yerr=stds, marker="o", capsize=3, linewidth=1.5)
        ax.set_xlabel(r"Mutation factor $m$")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(xs)
    axes[0].set_ylabel("Val Acc (%)")
    fig.suptitle("Multi-seed $m$-sweep (mean ± std)", fontsize=11)
    fig.tight_layout()
    out = LATEX / "fig_a1_m_sweep.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    # also copy next to MS figures
    ms_root = Path(__file__).resolve().parents[3]
    fig.savefig(ms_root / "fig_a1_m_sweep.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    main_df = pd.read_csv(RESULTS / "summary_main.csv")
    abl_df = pd.read_csv(RESULTS / "summary_ablation.csv")
    write_table3(main_df)
    write_ablation(abl_df)
    write_stats(load_runs())
    write_msweep_table_and_fig()
    print("Assets written to", LATEX)


if __name__ == "__main__":
    main()
