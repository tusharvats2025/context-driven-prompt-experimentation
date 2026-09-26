"""
generate_charts.py
==================
Produces all experiment figures from aggregate_summary.json.

Figures:
  figure1_score_by_pattern.png   — bar chart: avg score per pattern
  figure2_time_vs_validity.png   — scatter: time vs score
  figure3_failure_heatmap.png    — heatmap: metric × pattern failure rates
  figure4_cdp_vs_rest.png        — CDP vs avg of all others, per metric

CDP v1.0 — linear_regression domain
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
SUMMARY_PATH = PROJECT_ROOT / "analysis" / "outputs" / "aggregate_summary.json"
FIGURES_DIR  = PROJECT_ROOT / "analysis" / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family"     : "DejaVu Sans",
    "font.size"       : 11,
    "axes.spines.top" : False,
    "axes.spines.right": False,
    "axes.grid"       : True,
    "grid.alpha"      : 0.3,
    "figure.dpi"      : 150,
})

PATTERN_ORDER  = ["zero_shot", "few_shot", "chain_of_thought",
                  "role_based", "legacy_expert", "cdp"]
PATTERN_LABELS = ["Zero-shot", "Few-shot", "Chain-of-thought",
                  "Role-based", "Legacy Expert", "CDP"]

METRIC_ORDER   = [
    "no_dummy_trap", "vif_check", "heteroscedasticity_handled",
    "outliers_treated", "scaler_no_leakage", "residuals_checked", "model_saves",
]
METRIC_LABELS  = [
    "No Dummy Trap", "VIF Check", "Heteroscedasticity",
    "Outliers Treated", "Scaler No Leak", "Residuals Checked", "Model Saves",
]

# Color palette
COLORS = {
    "cdp"             : "#2563EB",   # blue — CDP stands out
    "others"          : "#94A3B8",   # slate
    "pass"            : "#22C55E",
    "fail"            : "#EF4444",
    "neutral"         : "#F59E0B",
}


def load_summary() -> dict:
    with open(SUMMARY_PATH) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Score by pattern (bar chart)
# ─────────────────────────────────────────────────────────────────────────────

def fig1_score_by_pattern(summary: dict) -> None:
    stats  = summary["pattern_stats"]
    scores = []
    stds   = []
    colors = []

    for p in PATTERN_ORDER:
        s = stats.get(p, {})
        scores.append(s.get("avg_score_pct", 0))
        stds.append(s.get("std_score_pct", 0))
        colors.append(COLORS["cdp"] if p == "cdp" else COLORS["others"])

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        PATTERN_LABELS, scores, color=colors,
        edgecolor="white", linewidth=1.2,
        yerr=stds, capsize=5, error_kw={"ecolor": "#64748B", "elinewidth": 1.5},
    )

    # Annotate bars
    for bar, score, std_val in zip(bars, scores, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std_val + 1.5,
            f"{score:.1f}%",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_ylim(0, 115)
    ax.set_ylabel("Average Score (% of 7 metrics passed)", fontsize=12)
    ax.set_title("Prompt Pattern Comparison — CDP vs Baseline Patterns",
                 fontsize=13, fontweight="bold", pad=14)

    # Legend
    cdp_patch   = mpatches.Patch(color=COLORS["cdp"],    label="CDP (proposed)")
    other_patch = mpatches.Patch(color=COLORS["others"], label="Baseline patterns")
    ax.legend(handles=[cdp_patch, other_patch], loc="upper left")

    # Hypothesis band
    ax.axhline(71, color=COLORS["cdp"], linestyle="--", alpha=0.4, linewidth=1)
    ax.text(5.5, 73, "CDP target ≥71%", color=COLORS["cdp"], fontsize=9, ha="right")

    plt.tight_layout()
    out = FIGURES_DIR / "figure1_score_by_pattern.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[Charts] → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Time vs Score scatter
# ─────────────────────────────────────────────────────────────────────────────

def fig2_time_vs_validity(summary: dict) -> None:
    stats   = summary["pattern_stats"]
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, p in enumerate(PATTERN_ORDER):
        s     = stats.get(p, {})
        score = s.get("avg_score_pct", 0)
        time_ = s.get("avg_time_sec", 0)
        color = COLORS["cdp"] if p == "cdp" else COLORS["others"]

        ax.scatter(time_, score, s=160, color=color, zorder=5,
                   edgecolors="white", linewidth=1.5)
        ax.annotate(
            PATTERN_LABELS[i],
            (time_, score),
            xytext=(6, 4), textcoords="offset points",
            fontsize=9,
        )

    ax.set_xlabel("Average Inference Time (seconds)", fontsize=12)
    ax.set_ylabel("Average Score (%)", fontsize=12)
    ax.set_title("Time vs Validity — CDP Upper-Right is the Target",
                 fontsize=13, fontweight="bold", pad=14)

    # Quadrant labels
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    mx   = (xlim[0] + xlim[1]) / 2
    my   = (ylim[0] + ylim[1]) / 2
    ax.text(mx * 1.4, my * 1.6, "High quality\nHigh cost",
            color="#64748B", fontsize=8, ha="center", alpha=0.7)
    ax.text(mx * 0.6, my * 1.6, "High quality\nLow cost",
            color="#22C55E", fontsize=8, ha="center", alpha=0.7)

    plt.tight_layout()
    out = FIGURES_DIR / "figure2_time_vs_validity.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[Charts] → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Failure heatmap (metric × pattern)
# ─────────────────────────────────────────────────────────────────────────────

def fig3_failure_heatmap(summary: dict) -> None:
    stats = summary["pattern_stats"]

    # Build matrix: rows=metrics, cols=patterns
    matrix = np.zeros((len(METRIC_ORDER), len(PATTERN_ORDER)))
    for j, pattern in enumerate(PATTERN_ORDER):
        s    = stats.get(pattern, {})
        rates= s.get("metric_pass_rates", {})
        for i, metric in enumerate(METRIC_ORDER):
            # Convert pass rate → fail rate
            matrix[i, j] = 100 - rates.get(metric, 0)

    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(matrix, cmap="RdYlGn_r", vmin=0, vmax=100, aspect="auto")

    # Annotate cells
    for i in range(len(METRIC_ORDER)):
        for j in range(len(PATTERN_ORDER)):
            val   = matrix[i, j]
            color = "white" if val > 60 or val < 20 else "black"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    ax.set_xticks(range(len(PATTERN_ORDER)))
    ax.set_xticklabels(PATTERN_LABELS, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(len(METRIC_ORDER)))
    ax.set_yticklabels(METRIC_LABELS, fontsize=10)
    ax.set_title("Failure Rate per Metric per Pattern (%)\n"
                 "Green = few failures (good), Red = many failures (bad)",
                 fontsize=12, fontweight="bold", pad=12)

    plt.colorbar(im, ax=ax, label="Failure Rate (%)", shrink=0.8)
    plt.tight_layout()

    out = FIGURES_DIR / "figure3_failure_heatmap.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[Charts] → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: CDP vs rest — per-metric bar comparison
# ─────────────────────────────────────────────────────────────────────────────

def fig4_cdp_vs_rest(summary: dict) -> None:
    stats    = summary["pattern_stats"]
    others   = [p for p in PATTERN_ORDER if p != "cdp"]

    cdp_rates  = []
    other_rates = []

    cdp_s = stats.get("cdp", {}).get("metric_pass_rates", {})

    for metric in METRIC_ORDER:
        cdp_rates.append(cdp_s.get(metric, 0))

        vals = [
            stats.get(p, {}).get("metric_pass_rates", {}).get(metric, 0)
            for p in others
            if p in stats
        ]
        other_rates.append(sum(vals) / len(vals) if vals else 0)

    x    = np.arange(len(METRIC_LABELS))
    w    = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))

    bars1 = ax.bar(x - w/2, cdp_rates,   w, label="CDP",            color=COLORS["cdp"],    alpha=0.9)
    bars2 = ax.bar(x + w/2, other_rates, w, label="Baseline (avg)", color=COLORS["others"], alpha=0.9)

    # Annotate
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f"{bar.get_height():.0f}%", ha="center", va="bottom",
                fontsize=8, color=COLORS["cdp"], fontweight="bold")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f"{bar.get_height():.0f}%", ha="center", va="bottom",
                fontsize=8, color="#475569")

    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_LABELS, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Pass Rate (%)", fontsize=11)
    ax.set_ylim(0, 115)
    ax.set_title("CDP vs Baseline Patterns — Per-Metric Pass Rate",
                 fontsize=13, fontweight="bold", pad=14)
    ax.legend(fontsize=10)

    plt.tight_layout()
    out = FIGURES_DIR / "figure4_cdp_vs_rest.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[Charts] → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not SUMMARY_PATH.exists():
        print("[Charts] aggregate_summary.json not found. Run aggregate_results.py first.")
        import sys; sys.exit(1)

    summary = load_summary()

    fig1_score_by_pattern(summary)
    fig2_time_vs_validity(summary)
    fig3_failure_heatmap(summary)
    fig4_cdp_vs_rest(summary)

    print("[Charts] All figures saved to:", FIGURES_DIR)
