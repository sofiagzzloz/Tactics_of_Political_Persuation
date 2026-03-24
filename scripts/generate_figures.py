"""
Generates all paper figures from the results folder.
Saves high-resolution PNGs to results/figures/.

Figures produced:
  fig1_per_label_f1.png        — grouped bar: per-label F1 for all 4 models
  fig2_overall_metrics.png     — bar: micro/macro F1 per model
  fig3_precision_recall.png    — grouped bar: P and R per label for each model
  fig4_temporal_trends.png     — line chart: label frequency by year
  fig5_speaker_heatmap.png     — heatmap: top 15 speakers x labels
  fig6_error_analysis.png      — stacked bar: FP and FN counts per label (DistilBERT)

Run:
    python scripts/generate_figures.py
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import f1_score, precision_score, recall_score

os.makedirs("results/figures", exist_ok=True)

LABEL_COLS = [
    "emotion_appeal",
    "authority_appeal",
    "polarization",
    "presumption",
    "exaggeration",
    "rhetorical_framing",
]
LABEL_SHORT = [
    "Emotion\nAppeal",
    "Authority\nAppeal",
    "Polarization",
    "Presumption",
    "Exaggeration",
    "Rhetorical\nFraming",
]

# Colour palette (one per model, colourblind-friendly)
MODEL_COLORS = {
    "Dummy":        "#b0b0b0",
    "TF-IDF + LR":  "#4878d0",
    "TF-IDF + SVM": "#ee854a",
    "DistilBERT":   "#6acc65",
}

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       11,
    "axes.titlesize":  13,
    "axes.labelsize":  11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":      150,
})


# Helpers
def latest(prefix):
    folders = sorted(glob.glob(f"results/{prefix}_*/"))
    if not folders:
        raise FileNotFoundError(f"No results/{prefix}_* folder found.")
    return folders[-1]


def per_label_metrics(preds_csv):
    full_df  = pd.read_csv("dataset/dataset_annotated_final.csv").dropna(subset=["text"])
    preds_df = pd.read_csv(preds_csv)
    merged   = preds_df.merge(
        full_df[["text"] + LABEL_COLS], on="text", how="left", suffixes=("_pred", "_true")
    )
    f1s, precs, recs = [], [], []
    for col in LABEL_COLS:
        pc = f"{col}_pred" if f"{col}_pred" in merged.columns else col
        tc = f"{col}_true" if f"{col}_true" in merged.columns else col
        if pc == tc:
            f1s.append(0); precs.append(0); recs.append(0)
            continue
        yt = merged[tc].fillna(0).astype(int).values
        yp = merged[pc].fillna(0).astype(int).values
        f1s.append(round(f1_score(yt, yp, zero_division=0), 4))
        precs.append(round(precision_score(yt, yp, zero_division=0), 4))
        recs.append(round(recall_score(yt, yp, zero_division=0), 4))
    return np.array(f1s), np.array(precs), np.array(recs)


# Load all model results
bert_dir      = latest("distilbert")
baselines_dir = latest("baselines")

all_models = {}

# DistilBERT
bert_preds = os.path.join(bert_dir, "test_predictions.csv")
with open(os.path.join(bert_dir, "test_metrics.json")) as f:
    bert_agg = json.load(f)
bert_f1, bert_p, bert_r = per_label_metrics(bert_preds)
all_models["DistilBERT"] = {
    "f1": bert_f1, "precision": bert_p, "recall": bert_r,
    "micro_f1": bert_agg.get("eval_f1_micro", bert_agg.get("micro_f1", 0)),
    "macro_f1": bert_agg.get("eval_f1_macro", bert_agg.get("macro_f1", 0)),
}

# Baselines
for key, display in [("dummy", "Dummy"), ("lr", "TF-IDF + LR"), ("svm", "TF-IDF + SVM")]:
    subdir = os.path.join(baselines_dir, key)
    if not os.path.exists(subdir):
        print(f"Skipping {display}: {subdir} not found")
        continue
    preds_csv = os.path.join(subdir, "test_predictions.csv")
    with open(os.path.join(subdir, "test_metrics.json")) as f:
        agg = json.load(f)
    f1, p, r = per_label_metrics(preds_csv)
    all_models[display] = {
        "f1": f1, "precision": p, "recall": r,
        "micro_f1": agg.get("micro_f1", 0),
        "macro_f1": agg.get("macro_f1", 0),
    }

model_names = ["Dummy", "TF-IDF + LR", "TF-IDF + SVM", "DistilBERT"]
model_names = [m for m in model_names if m in all_models]

print(f"Models loaded: {model_names}")



# FIG 1 — Per-label F1 comparison (grouped bar)
n_labels = len(LABEL_COLS)
n_models = len(model_names)
x        = np.arange(n_labels)
bar_w    = 0.8 / n_models
offsets  = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * bar_w

fig, ax = plt.subplots(figsize=(13, 5))

for i, mname in enumerate(model_names):
    f1s = all_models[mname]["f1"]
    bars = ax.bar(
        x + offsets[i], f1s,
        width=bar_w * 0.9,
        color=MODEL_COLORS[mname],
        label=mname,
        edgecolor="white",
        linewidth=0.5,
    )
    # value labels on bars
    for bar, val in zip(bars, f1s):
        if val > 0.04:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.2f}",
                ha="center", va="bottom",
                fontsize=7.5, color="#444444",
            )

ax.set_xticks(x)
ax.set_xticklabels(LABEL_SHORT, fontsize=10)
ax.set_ylabel("F1 Score")
ax.set_ylim(0, 1.05)
ax.set_title("Per-label F1 score by model")
ax.legend(loc="upper right", framealpha=0.9, fontsize=10)
ax.axhline(0, color="black", linewidth=0.5)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("results/figures/fig1_per_label_f1.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved fig1_per_label_f1.png")



# FIG 2 — Overall micro / macro F1
micro_vals = [all_models[m]["micro_f1"] for m in model_names]
macro_vals = [all_models[m]["macro_f1"] for m in model_names]

x2    = np.arange(len(model_names))
bar_w2 = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
b1 = ax.bar(x2 - bar_w2 / 2, micro_vals, bar_w2, label="Micro F1",
            color=[MODEL_COLORS[m] for m in model_names], edgecolor="white", linewidth=0.5)
b2 = ax.bar(x2 + bar_w2 / 2, macro_vals, bar_w2, label="Macro F1",
            color=[MODEL_COLORS[m] for m in model_names], edgecolor="white", linewidth=0.5,
            alpha=0.6, hatch="//")

for bars in [b1, b2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.3f}", ha="center", va="bottom", fontsize=9)

ax.set_xticks(x2)
ax.set_xticklabels(model_names, fontsize=10)
ax.set_ylabel("F1 Score")
ax.set_ylim(0, 0.85)
ax.set_title("Overall micro and macro F1 by model")
solid_patch  = mpatches.Patch(color="grey",         label="Micro F1")
hatch_patch  = mpatches.Patch(facecolor="grey", hatch="//", label="Macro F1", alpha=0.6)
ax.legend(handles=[solid_patch, hatch_patch], fontsize=10)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("results/figures/fig2_overall_metrics.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved fig2_overall_metrics.png")



# FIG 3 — Precision vs Recall per label (DistilBERT vs best baseline)
# Show LR and DistilBERT side by side; best baseline is SVM if available else LR
best_base = "TF-IDF + SVM" if "TF-IDF + SVM" in all_models else "TF-IDF + LR"
compare_pair = [best_base, "DistilBERT"]
compare_pair = [m for m in compare_pair if m in all_models]

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

for ax, mname in zip(axes, compare_pair):
    p_vals = all_models[mname]["precision"]
    r_vals = all_models[mname]["recall"]
    xpos   = np.arange(n_labels)
    bw     = 0.35
    ax.bar(xpos - bw / 2, p_vals, bw, label="Precision",
           color=MODEL_COLORS[mname], edgecolor="white")
    ax.bar(xpos + bw / 2, r_vals, bw, label="Recall",
           color=MODEL_COLORS[mname], edgecolor="white", alpha=0.55, hatch="//")
    ax.set_xticks(xpos)
    ax.set_xticklabels(LABEL_SHORT, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_title(mname)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    p_patch = mpatches.Patch(color=MODEL_COLORS[mname],           label="Precision")
    r_patch = mpatches.Patch(color=MODEL_COLORS[mname], alpha=0.55,
                              hatch="//", label="Recall")
    ax.legend(handles=[p_patch, r_patch], fontsize=9)

axes[0].set_ylabel("Score")
fig.suptitle("Precision vs Recall per label", fontsize=13)
plt.tight_layout()
plt.savefig("results/figures/fig3_precision_recall.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved fig3_precision_recall.png")



# FIG 4 — Temporal trends (label frequency by year)
year_csv = "results/label_by_year.csv"
if os.path.exists(year_csv):
    year_df = pd.read_csv(year_csv)
    year_df = year_df[year_df["year"] >= 1900]   # drop ancient/noisy years

    LINE_COLORS = {
        "emotion_appeal":    "#e15759",
        "authority_appeal":  "#4e79a7",
        "polarization":      "#f28e2b",
        "presumption":       "#76b7b2",
        "exaggeration":      "#b07aa1",
        "rhetorical_framing":"#59a14f",
    }

    fig, ax = plt.subplots(figsize=(13, 5))
    for col, short in zip(LABEL_COLS, LABEL_SHORT):
        ax.plot(
            year_df["year"], year_df[col],
            label=short.replace("\n", " "),
            color=LINE_COLORS[col],
            linewidth=1.8,
            marker="o", markersize=3, alpha=0.85,
        )

    ax.set_xlabel("Year")
    ax.set_ylabel("Mean label frequency")
    ax.set_title("Persuasion tactic frequency over time")
    ax.legend(loc="upper left", fontsize=9, ncol=2, framealpha=0.9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig("results/figures/fig4_temporal_trends.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved fig4_temporal_trends.png")
else:
    print(f"Skipping fig4: {year_csv} not found — run analyze_predictions.py first")



# FIG 5 — Speaker heatmap (top 15 by total tactic use)
speaker_csv = "results/label_by_speaker.csv"
if os.path.exists(speaker_csv):
    sp_df = pd.read_csv(speaker_csv)

    # keep only the _mean columns for each label
    mean_cols = [f"{c}_mean" for c in LABEL_COLS if f"{c}_mean" in sp_df.columns]

    if mean_cols:
        sp_df["total"] = sp_df[mean_cols].sum(axis=1)
        top15          = sp_df.nlargest(15, "total").set_index("speaker")[mean_cols]
        top15.columns  = [c.replace("_mean", "").replace("_", " ").title() for c in mean_cols]
    else:
        # fallback: label cols stored directly
        available = [c for c in LABEL_COLS if c in sp_df.columns]
        sp_df["total"] = sp_df[available].sum(axis=1)
        top15 = sp_df.nlargest(15, "total").set_index("speaker")[available]
        top15.columns = [c.replace("_", " ").title() for c in available]

    try:
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(
            top15.astype(float),
            annot=True, fmt=".2f", cmap="YlOrRd",
            linewidths=0.4, linecolor="#dddddd",
            cbar_kws={"label": "Mean label frequency"},
            ax=ax,
        )
        ax.set_title("Top 15 speakers by total tactic use", pad=14)
        ax.set_xlabel("")
        ax.set_ylabel("")
        plt.xticks(rotation=30, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig("results/figures/fig5_speaker_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("Saved fig5_speaker_heatmap.png")
    except ImportError:
        print("seaborn not installed — skipping heatmap. Run: pip install seaborn")
else:
    print(f"Skipping fig5: {speaker_csv} not found — run analyze_predictions.py first")



# FIG 6 — Error analysis: FP and FN per label (DistilBERT)
error_csv = "results/error_analysis.csv"
if os.path.exists(error_csv):
    err_df  = pd.read_csv(error_csv)
    fp_vals = []
    fn_vals = []
    for col in LABEL_COLS:
        sub = err_df[err_df["label"] == col]
        fp_vals.append(len(sub[sub["error_type"] == "FP"]))
        fn_vals.append(len(sub[sub["error_type"] == "FN"]))

    x6   = np.arange(n_labels)
    bw6  = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x6 - bw6 / 2, fp_vals, bw6, label="False Positives",
           color="#e15759", edgecolor="white")
    ax.bar(x6 + bw6 / 2, fn_vals, bw6, label="False Negatives",
           color="#4e79a7", edgecolor="white")

    for xp, fp, fn in zip(x6, fp_vals, fn_vals):
        if fp > 0:
            ax.text(xp - bw6 / 2, fp + 0.2, str(fp), ha="center", va="bottom", fontsize=9)
        if fn > 0:
            ax.text(xp + bw6 / 2, fn + 0.2, str(fn), ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x6)
    ax.set_xticklabels(LABEL_SHORT, fontsize=10)
    ax.set_ylabel("Count")
    ax.set_title("DistilBERT error analysis: false positives and false negatives per label")
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig("results/figures/fig6_error_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved fig6_error_analysis.png")
else:
    print(f"Skipping fig6: {error_csv} not found — run analyze_predictions.py first")

print("\nAll figures saved to results/figures/")