

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = ROOT / "results" / "stability" / "stability_scores_final.csv"
OUTPUT_DIR = ROOT / "results" / "statistical"

os.makedirs(OUTPUT_DIR, exist_ok=True)


df = pd.read_csv(INPUT_CSV)
df = df[df["status"] == "success"] 

proto = df[df["type"] == "prototypical"]  
bound = df[df["type"] == "boundary"]     

print(f"Loaded: {len(df)} songs  (proto={len(proto)}, boundary={len(bound)})\n")


def analyze(proto_data, bound_data, method_name):
    """
    Compare the two groups and show the results

    proto_data: CV scores for prototypical songs
    bound_data: CV scores for boundary songs
    """
    p_mean, p_std = proto_data.mean(), proto_data.std()
    b_mean, b_std = bound_data.mean(), bound_data.std()

    t_stat, p_val = stats.ttest_ind(proto_data, bound_data)

    pooled_std = np.sqrt((p_std**2 + b_std**2) / 2)
    cohens_d   = (b_mean - p_mean) / pooled_std

    pct_diff = (b_mean - p_mean) / p_mean * 100

    print(f"{method_name} ")
    print(f"  Prototypical : {p_mean:.3f} ± {p_std:.3f}")
    print(f"  Boundary     : {b_mean:.3f} ± {b_std:.3f}  ({pct_diff:+.1f}%)")
    print(f"  t={t_stat:.3f},  p={p_val:.3f}  {' significant' if p_val < 0.05 else 'this is not significant'}")
    print(f"  Cohen's d = {cohens_d:.3f}  ({'small' if abs(cohens_d)<0.5 else 'medium' if abs(cohens_d)<0.8 else 'large'})")
    print()

    return {"method": method_name, "p_mean": p_mean, "p_std": p_std,
            "b_mean": b_mean, "b_std": b_std,
            "t": t_stat, "p": p_val, "d": cohens_d, "pct": pct_diff}

results = [
    analyze(proto["LIME_CV"], bound["LIME_CV"], "LIME"),
    analyze(proto["IG_CV"],   bound["IG_CV"],   "IG"),
]


fig, axes = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle("XAI Stability: Prototypical vs Boundary Songs", fontsize=13, fontweight="bold")

for ax, r in zip(axes, results):

    means = [r["p_mean"], r["b_mean"]]
    stds  = [r["p_std"],  r["b_std"]]

    ax.bar([0, 1], means, yerr=stds, capsize=8, width=0.5,
           color=["#4C9BE8", "#E85C4C"], edgecolor="black")

    y_top = max(means) + max(stds) + 0.02
    ax.plot([0, 1], [y_top, y_top], color="black", linewidth=1)
    ax.text(0.5, y_top + 0.005,
            f"p={r['p']:.3f}{'*' if r['p']<0.05 else ' (n.s.)'}   d={r['d']:.2f}",
            ha="center", fontsize=9)

    ax.set_title(r["method"], fontsize=12, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Prototypical", "Boundary"])
    ax.set_ylabel("Coefficient of Variation (CV)")
    ax.set_ylim(0, y_top + 0.05)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig1_stability_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()
print(" fig1_stability_comparison.png saved")


fig, axes = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle("CV Score Distribution", fontsize=13, fontweight="bold")

for ax, col, method_name in zip(axes, ["LIME_CV", "IG_CV"], ["LIME", "IG"]):

    p_vals = proto[col].dropna().values
    b_vals = bound[col].dropna().values

    bp = ax.boxplot([p_vals, b_vals], tick_labels=["Prototypical", "Boundary"],
                    patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=2))
    bp["boxes"][0].set_facecolor("#4C9BE8")
    bp["boxes"][1].set_facecolor("#E85C4C")

    for i, (vals, xpos) in enumerate([(p_vals, 1), (b_vals, 2)]):
        jitter = np.random.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(xpos + jitter, vals, alpha=0.5, s=20,
                   color=["#4C9BE8", "#E85C4C"][i])

    ax.set_title(method_name, fontsize=12, fontweight="bold")
    ax.set_ylabel("Coefficient of Variation (CV)")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig2_boxplot.png"), dpi=150, bbox_inches="tight")
plt.close()
print(" fig2_boxplot.png saved")


summary_rows = []
for r in results:
    summary_rows += [
        {"Method": r["method"], "Group": "Prototypical", "Mean CV": round(r["p_mean"], 3), "SD": round(r["p_std"], 3)},
        {"Method": r["method"], "Group": "Boundary",     "Mean CV": round(r["b_mean"], 3), "SD": round(r["b_std"], 3)},
    ]
pd.DataFrame(summary_rows).to_csv(os.path.join(OUTPUT_DIR, "summary_table_v2.csv"), index=False)
print("summary_table.csv saved")

print("\n Output is ", OUTPUT_DIR)