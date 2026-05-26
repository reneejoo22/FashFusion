import os
import re
import pickle
import pandas as pd

from collections import Counter
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    ENGLISH_STOP_WORDS
)
from scipy.sparse import save_npz


base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, "articles.csv")

df = pd.read_csv(csv_path)


base_stopwords = set(ENGLISH_STOP_WORDS)

manual_domain_stopwords = {
    "hm",
    "article",
    "product",
    "products",
    "item",
    "items",
    "collection",
    "collections",
    "fashion",
    "look",
    "looks",
    "shop",
    "style",
    "styles"
}

custom_stopwords = set()

synonym_dict = {
    "tee": "tshirt",
    "trainer": "sneakers",
    "trainers": "sneakers",
    "grey": "gray",
    "burgundy": "red",
    "ivory": "white"
}

tpo_keywords = {
    "office": ["office", "work", "business", "professional", "formal"],
    "party": ["party", "date", "night", "club", "cocktail"],
    "wedding": ["wedding", "bridal", "prom"],
    "beach": ["beach", "vacation", "resort"],
    "travel": ["travel", "trip", "airport", "pack"],
    "school": ["school", "campus", "college"],
    "sport": ["sport", "gym", "running", "training"],
    "outdoor": ["outdoor", "hiking", "camping"],
    "home": ["home", "decor", "interior"],
    "casual": ["casual", "daily", "street"]
}


def normalize_raw_tokens(text):
    if pd.isna(text):
        return []

    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = []

    for word in text.split():
        if len(word) <= 1:
            continue

        word = synonym_dict.get(word, word)
        tokens.append(word)

    return tokens


def clean_text(text):
    if pd.isna(text):
        return ""

    stopwords = (
        base_stopwords
        | manual_domain_stopwords
        | custom_stopwords
    )

    tokens = normalize_raw_tokens(text)

    cleaned = [
        word for word in tokens
        if word not in stopwords
    ]

    return " ".join(cleaned)


def show_top_words(texts, top_n=50):
    counter = Counter()

    for text in texts:
        counter.update(normalize_raw_tokens(text))

    print("\nTOP WORDS")
    for word, count in counter.most_common(top_n):
        print(word, count)


def compare_cleaning(text):
    original_tokens = normalize_raw_tokens(text)
    cleaned_tokens = clean_text(text).split()

    removed_tokens = [
        word for word in original_tokens
        if word not in cleaned_tokens
    ]

    print("\n원본 토큰:")
    print(original_tokens)

    print("\n남은 토큰:")
    print(cleaned_tokens)

    print("\n삭제된 토큰:")
    print(removed_tokens)


def removed_word_stats(texts, top_n=100):
    removed_counter = Counter()

    for text in texts:
        original_tokens = normalize_raw_tokens(text)
        cleaned_tokens = set(clean_text(text).split())

        for word in original_tokens:
            if word not in cleaned_tokens:
                removed_counter[word] += 1

    print(f"\nREMOVED WORDS TOP {top_n}")
    for word, count in removed_counter.most_common(top_n):
        print(word, count)

    return removed_counter


def extract_tpo_features(text):
    tokens = set(text.split())
    result = {}

    for key, words in tpo_keywords.items():
        result[f"tpo_{key}"] = int(
            any(word in tokens for word in words)
        )

    return result


text_columns = [
    "prod_name",
    "product_type_name",
    "product_group_name",
    "graphical_appearance_name",
    "colour_group_name",
    "perceived_colour_value_name",
    "perceived_colour_master_name",
    "department_name",
    "index_name",
    "index_group_name",
    "section_name",
    "garment_group_name",
    "detail_desc"
]

existing_columns = [
    col for col in text_columns
    if col in df.columns
]

df["raw_text"] = (
    df[existing_columns]
    .fillna("")
    .astype(str)
    .agg(" ".join, axis=1)
)

raw_texts = df["raw_text"].tolist()

show_top_words(raw_texts, top_n=50)

print("\n전처리 전/후 샘플 비교")
compare_cleaning(raw_texts[0])

removed_counter = removed_word_stats(raw_texts, top_n=100)

df["cleaned_text"] = df["raw_text"].apply(clean_text)

df = df[
    df["cleaned_text"].str.len() > 0
].reset_index(drop=True)

tpo_df = (
    df["cleaned_text"]
    .apply(extract_tpo_features)
    .apply(pd.Series)
)

df = pd.concat(
    [df, tpo_df],
    axis=1
)

output_csv_path = os.path.join(
    base_dir,
    "hnm_text_features_final.csv"
)

removed_csv_path = os.path.join(
    base_dir,
    "hnm_removed_words.csv"
)

keep_cols = [
    "article_id",
    "product_code",
    "prod_name",
    "product_type_name",
    "product_group_name",
    "graphical_appearance_name",
    "colour_group_name",
    "perceived_colour_value_name",
    "perceived_colour_master_name",
    "department_name",
    "index_name",
    "index_group_name",
    "section_name",
    "garment_group_name",
    "detail_desc",
    "cleaned_text"
]

keep_cols = [
    col for col in keep_cols
    if col in df.columns
]

tpo_cols = [
    col for col in df.columns
    if col.startswith("tpo_")
]

df[
    keep_cols + tpo_cols
].to_csv(
    output_csv_path,
    index=False,
    encoding="utf-8-sig"
)

removed_df = pd.DataFrame(
    removed_counter.most_common(),
    columns=["removed_word", "count"]
)

removed_df.to_csv(
    removed_csv_path,
    index=False,
    encoding="utf-8-sig"
)

texts = df["cleaned_text"].tolist()

vectorizer = TfidfVectorizer(
    lowercase=True,
    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    stop_words="english",
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.8,
    max_features=20000,
    sublinear_tf=True,
    norm="l2"
)

X_tfidf = vectorizer.fit_transform(texts)

features = vectorizer.get_feature_names_out()

tfidf_path = os.path.join(
    base_dir,
    "hnm_tfidf_features_final.npz"
)

vectorizer_path = os.path.join(
    base_dir,
    "hnm_tfidf_vectorizer_final.pkl"
)

vocab_path = os.path.join(
    base_dir,
    "hnm_tfidf_vocabulary.csv"
)

save_npz(
    tfidf_path,
    X_tfidf
)

with open(vectorizer_path, "wb") as f:
    pickle.dump(vectorizer, f)

pd.DataFrame(
    {"feature": features}
).to_csv(
    vocab_path,
    index=False,
    encoding="utf-8-sig"
)

must_keep_words = [
    "black",
    "white",
    "leather",
    "bag",
    "top",
    "dress",
    "jeans",
    "skirt",
    "boots",
    "necklace",
    "sunglasses",
    "denim",
    "jacket",
    "suede",
    "blue",
    "red"
]

print("\n핵심 패션 단어 TF-IDF vocab 포함 여부")
for word in must_keep_words:
    print(word, word in features)

print()
print("H&M article 수:", len(df))
print("사용 컬럼:", existing_columns)
print("TF-IDF shape:", X_tfidf.shape)
print("Saved:", output_csv_path)
print("Saved:", removed_csv_path)
print("Saved:", tfidf_path)
print("Saved:", vectorizer_path)
print("Saved:", vocab_path)
print()
print(df[["article_id", "cleaned_text"] + tpo_cols].head())