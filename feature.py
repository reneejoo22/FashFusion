import os
import json
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
json_path = os.path.join(base_dir, "train_no_dup.json")

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)


base_stopwords = set(ENGLISH_STOP_WORDS)

manual_domain_stopwords = {
    "polyvore",
    "featuring",
    "browse",
    "related",
    "website",
    "official",
    "http",
    "https",
    "www",
    "com",
    "html",
    "utm",
    "source",
    "medium",
    "contest",
    "contestentry",
    "fashion",
    "look",
    "looks",
    "shop",
    "set"
}

custom_stopwords = {
    "style",
    "styles"
}

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

month_map = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12
}


def month_to_season(month):
    if month in [3, 4, 5]:
        return "spring"

    if month in [6, 7, 8]:
        return "summer"

    if month in [9, 10, 11]:
        return "fall"

    if month in [12, 1, 2]:
        return "winter"

    return "unknown"


def normalize_raw_tokens(text):
    if not text:
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
    if not text:
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


def extract_month(text):
    text = str(text).lower()

    for month_name, month_num in month_map.items():
        if month_name in text:
            return month_num

    return 0


def extract_tpo_features(text):
    tokens = set(text.split())
    result = {}

    for key, words in tpo_keywords.items():
        result[f"tpo_{key}"] = int(
            any(word in tokens for word in words)
        )

    return result


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


raw_texts = []

for outfit in data:
    raw = " ".join(
        [
            outfit.get("name", ""),
            outfit.get("desc", "")
        ]
        +
        [
            item.get("name", "")
            for item in outfit.get("items", [])
        ]
    )

    raw_texts.append(raw)


show_top_words(raw_texts, top_n=50)

print("\n전처리 전/후 샘플 비교")
compare_cleaning(raw_texts[0])

removed_counter = removed_word_stats(raw_texts, top_n=100)


samples = []

for outfit in data:
    raw_title = outfit.get("name", "")
    raw_desc = outfit.get("desc", "")

    title = clean_text(raw_title)
    desc = clean_text(raw_desc)

    item_names = []
    categories = []
    prices = []
    likes = []

    for item in outfit.get("items", []):
        item_name = clean_text(item.get("name", ""))

        if item_name:
            item_names.append(item_name)

        categories.append(
            str(item.get("categoryid", -1))
        )

        price = item.get("price", -1)

        if isinstance(price, (int, float)) and price > 0:
            prices.append(price)

        like = item.get("likes", 0)

        if isinstance(like, (int, float)):
            likes.append(like)

    combined_text = " ".join(
        [
            title,
            desc,
            " ".join(item_names)
        ]
    ).strip()

    if not combined_text:
        continue

    month = extract_month(
        raw_title + " " + raw_desc
    )

    season = month_to_season(month)

    sample = {
        "set_id": outfit.get("set_id", ""),
        "text": combined_text,
        "items": item_names,
        "categories": categories,
        "month": month,
        "season": season,
        "avg_price": (
            sum(prices) / len(prices)
            if prices else 0
        ),
        "avg_likes": (
            sum(likes) / len(likes)
            if likes else 0
        )
    }

    sample.update(
        extract_tpo_features(combined_text)
    )

    samples.append(sample)


df = pd.DataFrame(samples)

json_out = os.path.join(
    base_dir,
    "polyvore_text_features_final.json"
)

csv_out = os.path.join(
    base_dir,
    "polyvore_tpo_features.csv"
)

removed_csv_out = os.path.join(
    base_dir,
    "polyvore_removed_words.csv"
)

with open(json_out, "w", encoding="utf-8") as f:
    json.dump(
        samples,
        f,
        indent=2,
        ensure_ascii=False
    )

tpo_cols = [
    col for col in df.columns
    if col.startswith("tpo_")
]

df[
    [
        "set_id",
        "month",
        "season",
        "avg_price",
        "avg_likes"
    ]
    + tpo_cols
].to_csv(
    csv_out,
    index=False,
    encoding="utf-8-sig"
)

removed_df = pd.DataFrame(
    removed_counter.most_common(),
    columns=["removed_word", "count"]
)

removed_df.to_csv(
    removed_csv_out,
    index=False,
    encoding="utf-8-sig"
)


texts = df["text"].tolist()

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
    "polyvore_tfidf_features_final.npz"
)

vec_path = os.path.join(
    base_dir,
    "polyvore_tfidf_vectorizer_final.pkl"
)

vocab_path = os.path.join(
    base_dir,
    "polyvore_tfidf_vocabulary.csv"
)

save_npz(
    tfidf_path,
    X_tfidf
)

with open(vec_path, "wb") as f:
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
print("Outfits:", len(df))
print("TF-IDF shape:", X_tfidf.shape)
print("Saved:", json_out)
print("Saved:", csv_out)
print("Saved:", removed_csv_out)
print("Saved:", tfidf_path)
print("Saved:", vec_path)
print("Saved:", vocab_path)
print()
print(df.head())