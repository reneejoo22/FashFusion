import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from datasets import load_from_disk
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")

# ── 그래프 설정 ──────────────────────────────
plt.rcParams["figure.dpi"]      = 150
plt.rcParams["figure.figsize"]  = (12, 6)
plt.rcParams["axes.spines.top"]    = False
plt.rcParams["axes.spines.right"]  = False

# 한글 폰트 설정 (Windows)
try:
    plt.rcParams["font.family"] = "Malgun Gothic"
except:
    pass

SAVE_DIR = "eda_results"
os.makedirs(SAVE_DIR, exist_ok=True)

print("=" * 60)
print("FashFusion EDA 분석 시작")
print("=" * 60)

# ================================================
# PART 1. Polyvore EDA
# ================================================
print("\n[PART 1] Polyvore 데이터 분석")

dataset = load_from_disk("data/polyvore")
df = dataset["data"].to_pandas()

# outfit_id 추출
df["outfit_id"]  = df["item_ID"].str.split("_").str[0]
df["item_index"] = df["item_ID"].str.split("_").str[1]
df["text"]       = df["text"].fillna("")
df["category"]   = df["category"].fillna("")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["synthetic_text"] = (df["text"] + " " + df["category"]).apply(clean_text)
df["text_length"]    = df["synthetic_text"].apply(lambda x: len(x.split()))

print(f"  전체 아이템 수 : {len(df):,}")
print(f"  전체 outfit 수 : {df['outfit_id'].nunique():,}")
print(f"  카테고리 수    : {df['category'].nunique()}")
print(f"  결측값 (text)  : {(df['text']=='').sum()}")
print(f"  결측값 (category): {(df['category']=='').sum()}")

# ── 그래프 1. 카테고리 분포 ──────────────────
fig, ax = plt.subplots(figsize=(14, 6))
cat_counts = df["category"].value_counts().head(20)
bars = ax.barh(cat_counts.index[::-1], cat_counts.values[::-1],
               color=sns.color_palette("Blues_d", len(cat_counts)))
ax.set_title("Polyvore — 카테고리 분포 Top 20", fontsize=15, fontweight="bold", pad=15)
ax.set_xlabel("아이템 수", fontsize=12)
ax.set_ylabel("카테고리", fontsize=12)
for bar, val in zip(bars, cat_counts.values[::-1]):
    ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2,
            f"{val:,}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/01_polyvore_category_dist.png", bbox_inches="tight")
plt.close()
print("저장: 01_polyvore_category_dist.png")

# ── 그래프 2. outfit당 아이템 수 분포 ───────
fig, ax = plt.subplots(figsize=(10, 5))
outfit_sizes = df.groupby("outfit_id").size()
outfit_sizes.value_counts().sort_index().plot(
    kind="bar", ax=ax,
    color=sns.color_palette("Blues_d", outfit_sizes.value_counts().nunique())
)
ax.set_title("Polyvore — outfit당 아이템 수 분포", fontsize=15, fontweight="bold", pad=15)
ax.set_xlabel("아이템 수", fontsize=12)
ax.set_ylabel("outfit 수", fontsize=12)
ax.tick_params(axis="x", rotation=0)
for p in ax.patches:
    ax.annotate(f"{int(p.get_height()):,}",
                (p.get_x() + p.get_width()/2, p.get_height()),
                ha="center", va="bottom", fontsize=10)
stats_text = (f"평균: {outfit_sizes.mean():.1f}개\n"
              f"최소: {outfit_sizes.min()}개\n"
              f"최대: {outfit_sizes.max()}개")
ax.text(0.98, 0.95, stats_text, transform=ax.transAxes,
        ha="right", va="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5))
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/02_polyvore_outfit_size_dist.png", bbox_inches="tight")
plt.close()
print("저장: 02_polyvore_outfit_size_dist.png")

# ── 그래프 3. 텍스트 길이 분포 ──────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df["text_length"], bins=30, color="#3B82F6", edgecolor="white", alpha=0.85)
ax.axvline(df["text_length"].mean(), color="red", linestyle="--",
           linewidth=1.5, label=f"평균: {df['text_length'].mean():.1f}")
ax.axvline(df["text_length"].median(), color="orange", linestyle="--",
           linewidth=1.5, label=f"중앙값: {df['text_length'].median():.1f}")
ax.set_title("Polyvore — Synthetic Text 길이 분포 (단어 수)", fontsize=15, fontweight="bold", pad=15)
ax.set_xlabel("단어 수", fontsize=12)
ax.set_ylabel("빈도", fontsize=12)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/03_polyvore_text_length_dist.png", bbox_inches="tight")
plt.close()
print("저장: 03_polyvore_text_length_dist.png")

# ── 그래프 4. Positive / Negative 쌍 비율 ───
import random
from itertools import combinations

df_sample = df.sample(min(3000, len(df)), random_state=42).reset_index(drop=True)
df_sample["outfit_id_s"] = df_sample["item_ID"].str.split("_").str[0]
id_to_idx = {item_id: idx for idx, item_id in enumerate(df_sample["item_ID"])}
outfit_groups = df_sample.groupby("outfit_id_s")["item_ID"].apply(list)

pos_pairs, neg_pairs = [], []
for _, item_ids in outfit_groups.items():
    for id_a, id_b in combinations(item_ids, 2):
        if id_a in id_to_idx and id_b in id_to_idx:
            pos_pairs.append((id_to_idx[id_a], id_to_idx[id_b], 1))

all_ids = df_sample["item_ID"].tolist()
while len(neg_pairs) < len(pos_pairs):
    id_a, id_b = random.choice(all_ids), random.choice(all_ids)
    if id_a.split("_")[0] != id_b.split("_")[0]:
        if id_a in id_to_idx and id_b in id_to_idx:
            neg_pairs.append((id_to_idx[id_a], id_to_idx[id_b], 0))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 파이차트
labels  = ["Positive\n(실제 코디 쌍)", "Negative\n(랜덤 쌍)"]
sizes   = [len(pos_pairs), len(neg_pairs)]
colors  = ["#3B82F6", "#E5E7EB"]
explode = (0.05, 0)
axes[0].pie(sizes, labels=labels, colors=colors, explode=explode,
            autopct="%1.1f%%", startangle=90,
            textprops={"fontsize": 12})
axes[0].set_title("레이블 비율", fontsize=13, fontweight="bold")

# 바차트
bars = axes[1].bar(labels, sizes, color=colors, edgecolor="gray", width=0.4)
axes[1].set_title("레이블 수량", fontsize=13, fontweight="bold")
axes[1].set_ylabel("쌍 수", fontsize=12)
for bar, val in zip(bars, sizes):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                 f"{val:,}", ha="center", fontsize=12, fontweight="bold")

plt.suptitle("Polyvore — Positive / Negative 레이블 분포",
             fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/04_polyvore_label_dist.png", bbox_inches="tight")
plt.close()
print("저장: 04_polyvore_label_dist.png")

# ================================================
# PART 2. H&M EDA
# ================================================
print("\n[PART 2] H&M 데이터 분석")

articles     = pd.read_csv("data/hnm/articles.csv")
transactions = pd.read_csv("data/hnm/transactions_train.csv",
                           parse_dates=["t_dat"])

print(f"  articles:     {articles.shape[0]:,}개 상품")
print(f"  transactions: {transactions.shape[0]:,}개 거래")

# 결측값 처리
text_cols = ["prod_name", "product_type_name", "product_group_name",
             "colour_group_name", "garment_group_name",
             "section_name", "detail_desc"]
for col in text_cols:
    articles[col] = articles[col].fillna("")

articles["synthetic_text"] = (
    articles["prod_name"]          + " " +
    articles["product_type_name"]  + " " +
    articles["product_group_name"] + " " +
    articles["colour_group_name"]  + " " +
    articles["garment_group_name"] + " " +
    articles["section_name"]       + " " +
    articles["detail_desc"]
).apply(clean_text)
articles["text_length"] = articles["synthetic_text"].apply(lambda x: len(x.split()))

print(f"  결측값 (detail_desc): {(articles['detail_desc']=='').sum():,}")

# ── 그래프 5. 상품 타입 분포 Top 20 ─────────
fig, ax = plt.subplots(figsize=(14, 6))
type_counts = articles["product_type_name"].value_counts().head(20)
bars = ax.barh(type_counts.index[::-1], type_counts.values[::-1],
               color=sns.color_palette("Greens_d", len(type_counts)))
ax.set_title("H&M — 상품 타입 분포 Top 20", fontsize=15, fontweight="bold", pad=15)
ax.set_xlabel("상품 수", fontsize=12)
ax.set_ylabel("상품 타입", fontsize=12)
for bar, val in zip(bars, type_counts.values[::-1]):
    ax.text(bar.get_width() + 30, bar.get_y() + bar.get_height()/2,
            f"{val:,}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/05_hnm_product_type_dist.png", bbox_inches="tight")
plt.close()
print("저장: 05_hnm_product_type_dist.png")

# ── 그래프 6. 색상 분포 Top 15 ──────────────
fig, ax = plt.subplots(figsize=(12, 5))
color_counts = articles["colour_group_name"].value_counts().head(15)
palette = sns.color_palette("Set2", len(color_counts))
bars = ax.bar(color_counts.index, color_counts.values,
              color=palette, edgecolor="white")
ax.set_title("H&M — 색상 분포 Top 15", fontsize=15, fontweight="bold", pad=15)
ax.set_xlabel("색상", fontsize=12)
ax.set_ylabel("상품 수", fontsize=12)
ax.tick_params(axis="x", rotation=45)
for bar, val in zip(bars, color_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
            f"{val:,}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/06_hnm_color_dist.png", bbox_inches="tight")
plt.close()
print("저장: 06_hnm_color_dist.png")

# ── 그래프 7. 거래량 시계열 ──────────────────
fig, ax = plt.subplots(figsize=(14, 5))
daily_trans = transactions.groupby("t_dat").size().reset_index(name="count")
ax.plot(daily_trans["t_dat"], daily_trans["count"],
        color="#3B82F6", linewidth=1.2, alpha=0.8)
ax.fill_between(daily_trans["t_dat"], daily_trans["count"],
                alpha=0.15, color="#3B82F6")
ax.set_title("H&M — 날짜별 거래량 (2018~2020)", fontsize=15, fontweight="bold", pad=15)
ax.set_xlabel("날짜", fontsize=12)
ax.set_ylabel("거래 수", fontsize=12)
ax.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"{int(x):,}")
)

# 샘플링 범위 표시 (최근 30일)
max_date = transactions["t_dat"].max()
min_date = max_date - pd.Timedelta(days=30)
ax.axvspan(min_date, max_date, alpha=0.2, color="red",
           label=f"Phase 2 샘플 구간 (최근 30일)")
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/07_hnm_daily_transactions.png", bbox_inches="tight")
plt.close()
print("저장: 07_hnm_daily_transactions.png")

# ── 그래프 8. 상품 그룹 분포 ────────────────
fig, ax = plt.subplots(figsize=(12, 5))
group_counts = articles["product_group_name"].value_counts()
colors = sns.color_palette("Blues_d", len(group_counts))
bars = ax.bar(group_counts.index, group_counts.values,
              color=colors, edgecolor="white")
ax.set_title("H&M — 상품 그룹 분포", fontsize=15, fontweight="bold", pad=15)
ax.set_xlabel("상품 그룹", fontsize=12)
ax.set_ylabel("상품 수", fontsize=12)
ax.tick_params(axis="x", rotation=30)
for bar, val in zip(bars, group_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
            f"{val:,}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/08_hnm_product_group_dist.png", bbox_inches="tight")
plt.close()
print("저장: 08_hnm_product_group_dist.png")

# ── 그래프 9. H&M 텍스트 길이 분포 ─────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(articles["text_length"], bins=40,
        color="#10B981", edgecolor="white", alpha=0.85)
ax.axvline(articles["text_length"].mean(), color="red", linestyle="--",
           linewidth=1.5, label=f"평균: {articles['text_length'].mean():.1f}")
ax.axvline(articles["text_length"].median(), color="orange", linestyle="--",
           linewidth=1.5, label=f"중앙값: {articles['text_length'].median():.1f}")
ax.set_title("H&M — Synthetic Text 길이 분포 (단어 수)", fontsize=15, fontweight="bold", pad=15)
ax.set_xlabel("단어 수", fontsize=12)
ax.set_ylabel("빈도", fontsize=12)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/09_hnm_text_length_dist.png", bbox_inches="tight")
plt.close()
print("저장: 09_hnm_text_length_dist.png")

# ── 그래프 10. Co-purchase 레이블 비율 ──────
RECENT_DAYS  = 30
max_date     = transactions["t_dat"].max()
min_date     = max_date - pd.Timedelta(days=RECENT_DAYS)
trans_sample = transactions[transactions["t_dat"] >= min_date]

co_purchase  = trans_sample.groupby(
    ["customer_id", "t_dat"]
)["article_id"].apply(list)

n_positive = sum(
    len(list(combinations(items, 2)))
    for items in co_purchase if len(items) >= 2
)
n_negative = n_positive  # 동일하게 샘플링

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

labels  = ["Positive\n(함께 구매된 쌍)", "Negative\n(랜덤 쌍)"]
sizes   = [n_positive, n_negative]
colors  = ["#10B981", "#E5E7EB"]
explode = (0.05, 0)
axes[0].pie(sizes, labels=labels, colors=colors, explode=explode,
            autopct="%1.1f%%", startangle=90,
            textprops={"fontsize": 12})
axes[0].set_title("레이블 비율", fontsize=13, fontweight="bold")

bars = axes[1].bar(labels, sizes, color=colors, edgecolor="gray", width=0.4)
axes[1].set_title("레이블 수량", fontsize=13, fontweight="bold")
axes[1].set_ylabel("쌍 수", fontsize=12)
axes[1].yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"{int(x):,}")
)
for bar, val in zip(bars, sizes):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() * 1.01,
                 f"{val:,}", ha="center", fontsize=11, fontweight="bold")

plt.suptitle("H&M — Positive / Negative 레이블 분포 (최근 30일)",
             fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/10_hnm_label_dist.png", bbox_inches="tight")
plt.close()
print("저장: 10_hnm_label_dist.png")

# ================================================
# 최종 요약 출력
# ================================================
print("\n" + "=" * 60)
print("EDA 분석 완료 요약")
print("=" * 60)

print("\n[Polyvore]")
print(f"  전체 아이템 수    : {len(df):,}")
print(f"  전체 outfit 수    : {df['outfit_id'].nunique():,}")
print(f"  카테고리 수       : {df['category'].nunique()}")
print(f"  outfit 평균 크기  : {df.groupby('outfit_id').size().mean():.1f}개")
print(f"  텍스트 평균 길이  : {df['text_length'].mean():.1f} 단어")
print(f"  Positive 쌍 수    : {len(pos_pairs):,}")
print(f"  Negative 쌍 수    : {len(neg_pairs):,}")

print("\n[H&M]")
print(f"  전체 상품 수      : {len(articles):,}")
print(f"  전체 거래 수      : {len(transactions):,}")
print(f"  상품 타입 수      : {articles['product_type_name'].nunique()}")
print(f"  색상 종류 수      : {articles['colour_group_name'].nunique()}")
print(f"  텍스트 평균 길이  : {articles['text_length'].mean():.1f} 단어")
print(f"  Phase 2 샘플 기간 : 최근 {RECENT_DAYS}일")
print(f"  Positive 쌍 수    : {n_positive:,}")

print(f"\n그래프 저장 위치: {os.path.abspath(SAVE_DIR)}/")
print("  01_polyvore_category_dist.png")
print("  02_polyvore_outfit_size_dist.png")
print("  03_polyvore_text_length_dist.png")
print("  04_polyvore_label_dist.png")
print("  05_hnm_product_type_dist.png")
print("  06_hnm_color_dist.png")
print("  07_hnm_daily_transactions.png")
print("  08_hnm_product_group_dist.png")
print("  09_hnm_text_length_dist.png")
print("  10_hnm_label_dist.png")