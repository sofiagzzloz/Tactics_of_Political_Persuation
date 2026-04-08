import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import os


# =========================
# CONFIG
# =========================
INPUT_PATH = "../dataset/dataset_annotated_final.csv"
TEXT_COLUMN = "text"

LABEL_COLUMNS = [
    "emotion_appeal",
    "authority_appeal",
    "polarization",
    "presumption",
    "exaggeration",
    "rhetorical_framing"
]

OUTPUT_DIR = "../results/language_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# LOAD DATA
# =========================
df = pd.read_csv(INPUT_PATH)

print("Dataset shape:", df.shape)


# =========================
# TF-IDF CONTRAST FUNCTION
# =========================
def get_top_features_contrast(subset_texts, other_texts, ngram_range=(1, 1), top_k=15):

    if ngram_range == (1, 1):
        min_df = 5
    else:
        min_df = 2  # lower threshold for bigrams

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=ngram_range,
        max_features=5000,
        min_df=min_df
    )

    # Fit on subset
    try:
        X_subset = vectorizer.fit_transform(subset_texts)
    except ValueError:
        print(f"⚠️ Skipping ngram_range={ngram_range} due to low data")
        return []

    # Transform others using same vocabulary
    X_other = vectorizer.transform(other_texts)

    features = vectorizer.get_feature_names_out()

    # Mean TF-IDF scores
    subset_mean = X_subset.mean(axis=0).A1
    other_mean = X_other.mean(axis=0).A1

    # Contrast score
    scores = subset_mean - other_mean

    # Frequency (how often token appears in subset)
    freq = (X_subset > 0).sum(axis=0).A1

    # Combine
    data = list(zip(features, scores, freq))

    # Sort by contrast score
    data = sorted(data, key=lambda x: x[1], reverse=True)

    return data[:top_k]


# =========================
# MAIN ANALYSIS
# =========================
results = []

for label in LABEL_COLUMNS:
    subset = df[df[label] == 1]
    others = df[df[label] == 0]

    subset_texts = subset[TEXT_COLUMN].dropna().tolist()
    other_texts = others[TEXT_COLUMN].dropna().tolist()

    print(f"\nAnalyzing {label}: {len(subset_texts)} samples")

    if len(subset_texts) == 0 or len(other_texts) == 0:
        continue

    # --- Top words ---
    top_words = get_top_features_contrast(
        subset_texts,
        other_texts,
        ngram_range=(1, 1)
    )

    # --- Top bigrams ---
    top_bigrams = get_top_features_contrast(
        subset_texts,
        other_texts,
        ngram_range=(2, 2)
    )

    # Save results with rank
    for i, (word, score, freq) in enumerate(top_words, start=1):
        results.append({
            "label": label,
            "type": "word",
            "feature": word,
            "score": score,
            "frequency": int(freq),
            "rank": i
        })

    for i, (phrase, score, freq) in enumerate(top_bigrams, start=1):
        results.append({
            "label": label,
            "type": "bigram",
            "feature": phrase,
            "score": score,
            "frequency": int(freq),
            "rank": i
        })


# =========================
# SAVE
# =========================
results_df = pd.DataFrame(results)
output_path = os.path.join(OUTPUT_DIR, "top_language_features.csv")

results_df.to_csv(output_path, index=False)

print(f"\nSaved to: {output_path}")