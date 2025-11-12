import os
import re
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from keras.layers import TextVectorization

# ==== reproducibility ====
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ==== paths ====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_PATH = os.path.join(DATA_DIR, "IMDB Dataset.csv")

# ==== config ====
TOP_WORDS = 10_000                          # vocabulary cap (includes special tokens)
SEQ_LENS = [25, 50, 100]                    # required output lengths
STD_FUNC = "lower_and_strip_punctuation"    # built-in standardization in Keras 3


def clean_text(text: str) -> str:
    """Light extra cleaning (HTML, non-letters) on top of Keras standardization."""
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)         # remove HTML tags
    text = re.sub(r"[^a-zA-Z']", " ", text)    # keep letters & apostrophes
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_base_vectorizer(x_train: pd.Series) -> TextVectorization:
    """
    Adapt a TextVectorization layer ONCE without fixed sequence length to learn the vocab.
    We'll reuse its vocabulary for all target sequence lengths to keep runs comparable.
    """
    vec = TextVectorization(
        max_tokens=TOP_WORDS,
        standardize=STD_FUNC,
        output_mode="int",         # integer token ids
    )
    vec.adapt(x_train.values)
    return vec


def vectorize_with_length(vocabulary, seq_len: int) -> TextVectorization:
    """
    Create a TextVectorization that uses a fixed vocabulary and pads/truncates to seq_len.
    Padding token id will be 0, OOV will be 1 (Keras convention).
    """
    return TextVectorization(
        max_tokens=TOP_WORDS,
        standardize=STD_FUNC,
        output_mode="int",
        output_sequence_length=seq_len,
        vocabulary=vocabulary,     # reuse the learned vocab
    )


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Loading data…")
    df = pd.read_csv(DATA_PATH)
    # Expect columns: review, sentiment (positive/negative)
    df["review"] = df["review"].astype(str).apply(clean_text)
    df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0}).astype(int)

    # Keep the predefined 50/50 split size from the assignment
    X_train, X_test, y_train, y_test = train_test_split(
        df["review"],
        df["sentiment"],
        test_size=0.5,
        random_state=SEED,
        stratify=df["sentiment"],
    )

    # ---- Learn vocab once on training data
    print("Adapting vectorizer to build vocabulary…")
    base_vec = build_base_vectorizer(X_train)
    vocabulary = base_vec.get_vocabulary()  # list: index -> token (0: pad, 1: [UNK])

    # Save vocab for reproducibility / inspection
    vocab_path = os.path.join(DATA_DIR, "vocab.txt")
    with open(vocab_path, "w", encoding="utf-8") as f:
        for tok in vocabulary:
            f.write(tok + "\n")
    print(f"Saved vocabulary ({len(vocabulary)} tokens) -> {vocab_path}")

    # ---- Vectorize and save for each required sequence length
    for seq_len in SEQ_LENS:
        print(f"Vectorizing with sequence length = {seq_len} …")
        vec = vectorize_with_length(vocabulary, seq_len)

        X_train_ids = vec(X_train.values).numpy().astype(np.int32)
        X_test_ids = vec(X_test.values).numpy().astype(np.int32)

        out_path = os.path.join(DATA_DIR, f"imdb_{seq_len}.npz")
        np.savez_compressed(
            out_path,
            X_train=X_train_ids,
            y_train=y_train.values.astype(np.int32),
            X_test=X_test_ids,
            y_test=y_test.values.astype(np.int32),
            pad_id=np.array([0], dtype=np.int32),
            unk_id=np.array([1], dtype=np.int32),
        )
        print(f"  -> Saved {out_path}  (X_train {X_train_ids.shape}, X_test {X_test_ids.shape})")

    # ---- Simple dataset stats for your report
    avg_len = df["review"].str.split().map(len).mean()
    print(f"\nDone. Avg raw review length (tokens, pre-vectorization): {avg_len:.1f}")
    print("Preprocessing complete. Files saved in /data/:")
    for L in SEQ_LENS:
        print(f"  - imdb_{L}.npz")


if __name__ == "__main__":
    main()
