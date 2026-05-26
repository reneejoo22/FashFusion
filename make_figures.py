"""
보고서용 캡처 이미지 생성
  - figure1_baseline_metrics.png  : 베이스라인(DistilBERT) 성능 지표
  - figure2_comparison.png        : 베이스라인 vs 제안 모델 비교
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 실험 수치 ──────────────────────────────────────────────
METRICS = {
    "labels":      ["Precision@10", "NMI", "Silhouette", "Davies-Bouldin*"],
    "baseline":    [0.2815,          0.0696, 0.0397,      3.5483],
    "proposed":    [0.3575,          0.0893, 0.0239,      6.1588],
    "higher_better":[True,           True,   True,        False],   # DB는 낮을수록 good
}

SIM_DATA = {
    "same_tpo":  {"baseline": 0.8943, "proposed": 0.0480},
    "diff_tpo":  {"baseline": 0.8880, "proposed": 0.0310},
    "gap":       {"baseline": 0.0062, "proposed": 0.0170},
}

TIME_DATA = {"baseline": 3.3, "proposed": 7.8}

COLORS = {
    "baseline": "#E07B54",   # 주황
    "proposed": "#4C72B0",   # 파랑
    "bg":       "#F7F7F7",
    "header":   "#2C3E50",
    "good":     "#27AE60",
    "bad":      "#E74C3C",
}


# ═══════════════════════════════════════════════════════════
# Figure 1 : 베이스라인(DistilBERT) 단독 성능 지표
# ═══════════════════════════════════════════════════════════
fig1, axes = plt.subplots(1, 3, figsize=(14, 5))
fig1.patch.set_facecolor(COLORS["bg"])
fig1.suptitle("Baseline Model (DistilBERT, 768-dim) — Performance Metrics",
              fontsize=14, fontweight="bold", color=COLORS["header"], y=1.01)

# ── (a) 성능 지표 가로 막대 ──
ax = axes[0]
ax.set_facecolor("white")
metric_names = ["Precision@10 ↑", "NMI ↑", "Silhouette ↑"]
vals         = [0.2815,             0.0696,   0.0397]
bar_colors   = [COLORS["baseline"]] * 3

bars = ax.barh(metric_names, vals, color=bar_colors, height=0.45, edgecolor="white")
for bar, v in zip(bars, vals):
    ax.text(v + 0.003, bar.get_y() + bar.get_height()/2,
            f"{v:.4f}", va="center", fontsize=10, color=COLORS["header"])
ax.set_xlim(0, 0.45)
ax.set_title("(a) Intrinsic Evaluation Scores", fontsize=11, pad=8)
ax.spines[["top","right","left"]].set_visible(False)
ax.tick_params(axis="y", labelsize=10)
ax.tick_params(axis="x", labelsize=9)
ax.set_xlabel("Score", fontsize=9)

# ── (b) 코사인 유사도 분포 ──
ax = axes[1]
ax.set_facecolor("white")
categories = ["Same-TPO\nSimilarity", "Diff-TPO\nSimilarity", "Gap\n(Separability)"]
vals2 = [SIM_DATA["same_tpo"]["baseline"],
         SIM_DATA["diff_tpo"]["baseline"],
         SIM_DATA["gap"]["baseline"]]
bar_c = [COLORS["baseline"], COLORS["baseline"], COLORS["bad"]]
bars2 = ax.bar(categories, vals2, color=bar_c, width=0.45, edgecolor="white")
for bar, v in zip(bars2, vals2):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
            f"{v:.4f}", ha="center", fontsize=10, color=COLORS["header"])
ax.set_ylim(0, 1.1)
ax.set_title("(b) Cosine Similarity Distribution", fontsize=11, pad=8)
ax.spines[["top","right"]].set_visible(False)
ax.tick_params(labelsize=10)
ax.set_ylabel("Cosine Similarity", fontsize=9)
ax.axhline(0.88, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
ax.text(2.35, 0.89, "collapse zone", fontsize=8, color="gray")

# ── (c) 요약 테이블 ──
ax = axes[2]
ax.set_facecolor("white")
ax.axis("off")
table_data = [
    ["Metric",               "Value",   "Note"],
    ["Precision@10",         "0.2815",  "vs random 0.10"],
    ["NMI (TPO GT)",         "0.0696",  "low alignment"],
    ["Silhouette",           "0.0397",  "near 0"],
    ["Davies-Bouldin",       "3.5483",  "lower is better"],
    ["Same-TPO Sim.",        "0.8943",  "—"],
    ["Diff-TPO Sim.",        "0.8880",  "—"],
    ["Sim. Gap",             "0.0062",  "very small"],
    ["Embed. Time (GPU)",    "3.3 s",   "RTX 5070 Ti"],
]
col_colors = [[COLORS["header"], COLORS["header"], COLORS["header"]]]
cell_colors = [["#ECECEC", "#ECECEC", "#ECECEC"]] * (len(table_data)-1)

tbl = ax.table(
    cellText=table_data[1:],
    colLabels=table_data[0],
    cellLoc="center",
    loc="center",
    cellColours=cell_colors,
    colColours=[COLORS["header"]]*3
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.1, 1.45)
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_text_props(color="white", fontweight="bold")
    cell.set_edgecolor("white")
ax.set_title("(c) Summary Table", fontsize=11, pad=8)

plt.tight_layout()
path1 = os.path.join(BASE_DIR, "figure1_baseline_metrics.png")
fig1.savefig(path1, dpi=180, bbox_inches="tight", facecolor=COLORS["bg"])
plt.close(fig1)
print(f"Saved: {path1}")


# ═══════════════════════════════════════════════════════════
# Figure 2 : 베이스라인 vs 제안 모델 비교
# ═══════════════════════════════════════════════════════════
fig2 = plt.figure(figsize=(16, 9))
fig2.patch.set_facecolor(COLORS["bg"])
gs = GridSpec(2, 3, figure=fig2, hspace=0.45, wspace=0.35)

fig2.suptitle("Baseline (DistilBERT) vs Proposed (TF-IDF + TruncatedSVD) — Comparison",
              fontsize=14, fontweight="bold", color=COLORS["header"])

labels_short = ["Baseline\n(DistilBERT)", "Proposed\n(TF-IDF+SVD)"]
bar_colors   = [COLORS["baseline"], COLORS["proposed"]]

# ── (a) Precision@10 ──
ax = fig2.add_subplot(gs[0, 0])
ax.set_facecolor("white")
vals = [0.2815, 0.3575]
bars = ax.bar(labels_short, vals, color=bar_colors, width=0.45, edgecolor="white")
for bar, v in zip(bars, vals):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.008, f"{v:.4f}",
            ha="center", fontsize=11, fontweight="bold", color=COLORS["header"])
ax.set_ylim(0, 0.50)
ax.set_title("Precision@10  ↑", fontsize=11, fontweight="bold")
ax.spines[["top","right"]].set_visible(False)
improvement = (vals[1]-vals[0])/vals[0]*100
ax.annotate(f"+{improvement:.1f}%", xy=(1, vals[1]), xytext=(1.3, vals[1]+0.03),
            arrowprops=dict(arrowstyle="->", color=COLORS["good"]),
            fontsize=10, color=COLORS["good"], fontweight="bold")

# ── (b) NMI ──
ax = fig2.add_subplot(gs[0, 1])
ax.set_facecolor("white")
vals = [0.0696, 0.0893]
bars = ax.bar(labels_short, vals, color=bar_colors, width=0.45, edgecolor="white")
for bar, v in zip(bars, vals):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.001, f"{v:.4f}",
            ha="center", fontsize=11, fontweight="bold", color=COLORS["header"])
ax.set_ylim(0, 0.14)
ax.set_title("NMI (TPO Label Alignment)  ↑", fontsize=11, fontweight="bold")
ax.spines[["top","right"]].set_visible(False)
improvement = (vals[1]-vals[0])/vals[0]*100
ax.annotate(f"+{improvement:.1f}%", xy=(1, vals[1]), xytext=(1.3, vals[1]+0.01),
            arrowprops=dict(arrowstyle="->", color=COLORS["good"]),
            fontsize=10, color=COLORS["good"], fontweight="bold")

# ── (c) Similarity Gap ──
ax = fig2.add_subplot(gs[0, 2])
ax.set_facecolor("white")
vals = [0.0062, 0.0170]
bars = ax.bar(labels_short, vals, color=bar_colors, width=0.45, edgecolor="white")
for bar, v in zip(bars, vals):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.0003, f"{v:.4f}",
            ha="center", fontsize=11, fontweight="bold", color=COLORS["header"])
ax.set_ylim(0, 0.028)
ax.set_title("Cosine Similarity Gap  ↑", fontsize=11, fontweight="bold")
ax.spines[["top","right"]].set_visible(False)
improvement = (vals[1]-vals[0])/vals[0]*100
ax.annotate(f"+{improvement:.1f}%\n(×2.7)", xy=(1, vals[1]), xytext=(1.3, vals[1]+0.003),
            arrowprops=dict(arrowstyle="->", color=COLORS["good"]),
            fontsize=10, color=COLORS["good"], fontweight="bold")

# ── (d) Davies-Bouldin (낮을수록 good) ──
ax = fig2.add_subplot(gs[1, 0])
ax.set_facecolor("white")
vals = [3.5483, 6.1588]
bars = ax.bar(labels_short, vals, color=bar_colors, width=0.45, edgecolor="white")
for bar, v in zip(bars, vals):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.05, f"{v:.4f}",
            ha="center", fontsize=11, fontweight="bold", color=COLORS["header"])
ax.set_ylim(0, 8.5)
ax.set_title("Davies-Bouldin Index  ↓\n(lower = better cluster shape)", fontsize=10, fontweight="bold")
ax.spines[["top","right"]].set_visible(False)
ax.annotate("Baseline\nbetter here", xy=(0, vals[0]), xytext=(-0.35, vals[0]+0.6),
            arrowprops=dict(arrowstyle="->", color=COLORS["bad"]),
            fontsize=9, color=COLORS["bad"])

# ── (e) 코사인 유사도 분리 시각화 ──
ax = fig2.add_subplot(gs[1, 1])
ax.set_facecolor("white")
x = np.arange(2)
w = 0.3
same = [SIM_DATA["same_tpo"]["baseline"], SIM_DATA["same_tpo"]["proposed"]]
diff = [SIM_DATA["diff_tpo"]["baseline"], SIM_DATA["diff_tpo"]["proposed"]]
b1 = ax.bar(x - w/2, same, width=w, color=bar_colors, alpha=0.9, label="Same TPO", edgecolor="white")
b2 = ax.bar(x + w/2, diff, width=w, color=bar_colors, alpha=0.45, label="Diff TPO",
            edgecolor="white", hatch="//")
ax.set_xticks(x)
ax.set_xticklabels(["Baseline\n(DistilBERT)", "Proposed\n(TF-IDF+SVD)"], fontsize=9)
ax.set_ylim(0, 1.1)
ax.set_title("Same vs Diff TPO Similarity", fontsize=11, fontweight="bold")
ax.spines[["top","right"]].set_visible(False)
ax.legend(fontsize=8, loc="upper right")
for bar, v in zip(list(b1)+list(b2), same+diff):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.01, f"{v:.3f}",
            ha="center", fontsize=8, color=COLORS["header"])

# ── (f) 종합 비교 테이블 ──
ax = fig2.add_subplot(gs[1, 2])
ax.set_facecolor("white")
ax.axis("off")
rows = [
    ["Metric",          "Baseline\n(DistilBERT)", "Proposed\n(TF-IDF+SVD)", "Winner"],
    ["Precision@10 ↑",  "0.2815",                 "0.3575",                  "Proposed"],
    ["NMI ↑",           "0.0696",                 "0.0893",                  "Proposed"],
    ["Sim. Gap ↑",      "0.0062",                 "0.0170",                  "Proposed"],
    ["Silhouette ↑",    "0.0397",                 "0.0239",                  "Baseline"],
    ["D-B Index ↓",     "3.5483",                 "6.1588",                  "Baseline"],
    ["Embed Time",      "3.3s (GPU)",             "7.8s (CPU)",              "Baseline"],
]
cell_text  = [r[1:] for r in rows[1:]]
col_labels = rows[0][1:]
row_labels = [r[0] for r in rows[1:]]

winner_colors = []
for r in rows[1:]:
    if r[-1] == "Proposed":
        winner_colors.append(["white", "white", "#D5F5E3"])
    else:
        winner_colors.append(["white", "white", "#FADBD8"])

tbl = ax.table(
    cellText=cell_text,
    rowLabels=row_labels,
    colLabels=col_labels,
    cellLoc="center",
    rowLoc="right",
    loc="center",
    cellColours=winner_colors,
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
tbl.scale(1.05, 1.55)
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_facecolor(COLORS["header"])
        cell.set_text_props(color="white", fontweight="bold")
    cell.set_edgecolor("white")
ax.set_title("(f) Head-to-Head Summary", fontsize=11, fontweight="bold", pad=10)

legend_patches = [
    mpatches.Patch(color=COLORS["baseline"], label="Baseline (DistilBERT)"),
    mpatches.Patch(color=COLORS["proposed"], label="Proposed (TF-IDF+SVD)"),
]
fig2.legend(handles=legend_patches, loc="lower center", ncol=2, fontsize=10,
            frameon=False, bbox_to_anchor=(0.5, -0.02))

path2 = os.path.join(BASE_DIR, "figure2_comparison.png")
fig2.savefig(path2, dpi=180, bbox_inches="tight", facecolor=COLORS["bg"])
plt.close(fig2)
print(f"Saved: {path2}")
print("Done.")
