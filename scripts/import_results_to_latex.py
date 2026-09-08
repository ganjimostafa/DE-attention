#!/usr/bin/env python3
"""Generate LaTeX table snippets from Phase 3 Colab CSV exports.

Usage (after copying results from Drive):
    python import_results_to_latex.py --results-dir ./results

Outputs:
    latex/table3_main.tex
    latex/table_a1_ablation.tex
    latex/table_a5_stats.tex
    latex/fig_a1_m_sweep.csv  (for plotting in MS)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def fmt_pm(mean: float, std: float, scale: float = 100.0, digits: int = 2) -> str:
    return f"{mean * scale:.{digits}f} $\\pm$ {std * scale:.{digits}f}"


def load_summary(results_dir: Path, name: str) -> pd.DataFrame:
    path = results_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run Colab notebook first.")
    return pd.read_csv(path)


def write_table3(df: pd.DataFrame, out: Path) -> None:
    """Main results: mean±std Acc and F1 macro per dataset/model."""
    lines = [
        "% Auto-generated from summary_main.csv — paste into Manuscript_body.tex",
        "\\begin{table}[H]",
        "  \\centering",
        "  \\small",
        "  \\caption{Multi-seed test results (mean $\\pm$ std).}",
        "  \\label{tab:multiseed_main}",
        "  \\begin{tabular}{llcc}",
        "    \\toprule",
        "    \\textbf{Dataset} & \\textbf{Model} & \\textbf{Acc (\\%)} & \\textbf{F1 macro (\\%)} \\\\",
        "    \\midrule",
    ]
    model_order = ["standard", "de_full", "de_best", "jde", "diff"]
    model_labels = {
        "standard": "Standard",
        "de_full": "DEAttention",
        "de_best": "DE/best/1/bin",
        "jde": "jDE",
        "diff": "DiffTransformer",
    }
    for dataset in sorted(df["dataset"].unique()):
        ds_rows = df[df["dataset"] == dataset]
        first = True
        for model in model_order:
            row = ds_rows[ds_rows["model"] == model]
            if row.empty:
                continue
            r = row.iloc[0]
            ds_cell = f"\\multirow{{{len(model_order)}}}{{*}}{{{dataset.upper()}}}" if first else ""
            first = False
            lines.append(
                f"    {ds_cell} & {model_labels.get(model, model)} "
                f"& {fmt_pm(r['test_acc_mean'], r['test_acc_std'])} "
                f"& {fmt_pm(r['test_f1_macro_mean'], r['test_f1_macro_std'])} \\\\"
            )
        lines.append("    \\midrule")
    if lines[-1].endswith("\\midrule"):
        lines[-1] = "    \\bottomrule"
    else:
        lines.append("    \\bottomrule")
    lines += ["  \\end{tabular}", "\\end{table}", ""]
    out.write_text("\n".join(lines), encoding="utf-8")


def write_ablation(df: pd.DataFrame, out: Path) -> None:
    lines = [
        "% Auto-generated from summary_ablation.csv",
        "\\begin{table}[H]",
        "  \\centering",
        "  \\small",
        "  \\caption{Ablation study (mean $\\pm$ std test accuracy, \\%).}",
        "  \\label{tab:ablation_a1}",
        "  \\begin{tabular}{llc}",
        "    \\toprule",
        "    \\textbf{Dataset} & \\textbf{Variant} & \\textbf{Acc (\\%)} \\\\",
        "    \\midrule",
    ]
    for _, r in df.sort_values(["dataset", "model"]).iterrows():
        lines.append(
            f"    {r['dataset'].upper()} & {r['model']} "
            f"& {fmt_pm(r['test_acc_mean'], r['test_acc_std'])} \\\\"
        )
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""]
    out.write_text("\n".join(lines), encoding="utf-8")


def write_stats(df: pd.DataFrame, out: Path) -> None:
    lines = [
        "% Auto-generated from stats_paired_ttest.csv",
        "\\begin{table}[H]",
        "  \\centering",
        "  \\small",
        "  \\caption{Paired $t$-tests: DEAttention vs baselines (test accuracy).}",
        "  \\label{tab:stats_a5}",
        "  \\begin{tabular}{llcc}",
        "    \\toprule",
        "    \\textbf{Dataset} & \\textbf{Comparison} & \\textbf{$p$-value} & \\textbf{95\\% CI} \\\\",
        "    \\midrule",
    ]
    for _, r in df.iterrows():
        ci = f"[{r['ci_low']*100:.2f}, {r['ci_high']*100:.2f}]"
        lines.append(
            f"    {r['dataset'].upper()} & {r['comparison']} "
            f"& {r['p_value']:.4f} & {ci} \\\\"
        )
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""]
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 CSV → LaTeX snippets")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Folder with summary_*.csv from Colab",
    )
    args = parser.parse_args()
    latex_dir = args.results_dir / "latex"
    latex_dir.mkdir(parents=True, exist_ok=True)

    main_csv = args.results_dir / "summary_main.csv"
    if main_csv.exists():
        write_table3(load_summary(args.results_dir, "summary_main.csv"), latex_dir / "table3_main.tex")
        print(f"Wrote {latex_dir / 'table3_main.tex'}")

    abl_csv = args.results_dir / "summary_ablation.csv"
    if abl_csv.exists():
        write_ablation(load_summary(args.results_dir, "summary_ablation.csv"), latex_dir / "table_a1_ablation.tex")
        print(f"Wrote {latex_dir / 'table_a1_ablation.tex'}")

    stats_csv = args.results_dir / "stats_paired_ttest.csv"
    if stats_csv.exists():
        write_stats(load_summary(args.results_dir, "stats_paired_ttest.csv"), latex_dir / "table_a5_stats.tex")
        print(f"Wrote {latex_dir / 'table_a5_stats.tex'}")

    ms_csv = args.results_dir / "summary_m_sweep.csv"
    if ms_csv.exists():
        import shutil
        shutil.copy(ms_csv, latex_dir / "fig_a1_m_sweep.csv")
        print(f"Copied {latex_dir / 'fig_a1_m_sweep.csv'}")

    if not any(args.results_dir.glob("summary_*.csv")):
        print("No summary CSVs found. Run pure_models.ipynb on Colab first.")


if __name__ == "__main__":
    main()
