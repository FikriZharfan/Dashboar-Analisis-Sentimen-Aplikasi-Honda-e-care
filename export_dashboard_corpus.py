"""
Bangun models/dashboard_corpus.csv + perbarui source_csv_mtime di metadata.json
tanpa melatih ulang model. Jalankan sekali jika dashboard lama di preprocessing.

Usage (dari folder sentiment-analysis):
    python export_dashboard_corpus.py
"""

from __future__ import annotations

import json
import os

import pandas as pd

from train_model import DATA_PATH, MODELS_DIR, preprocess_text, read_dataset


def main() -> None:
    df = read_dataset(DATA_PATH)
    df = df.dropna(subset=["content", "label"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["content"]).reset_index(drop=True)
    print(f"Memproses {len(df)} baris (bisa beberapa menit)...")
    df["cleaned_text"] = df["content"].astype(str).apply(preprocess_text)
    df = df[df["cleaned_text"].str.strip() != ""].reset_index(drop=True)

    corpus_path = os.path.join(MODELS_DIR, "dashboard_corpus.csv")
    df[["content", "label", "cleaned_text"]].to_csv(
        corpus_path, index=False, encoding="utf-8"
    )
    print(f"Tersimpan: {corpus_path} ({len(df)} baris)")

    meta_path = os.path.join(MODELS_DIR, "metadata.json")
    mtime = os.path.getmtime(DATA_PATH) if os.path.exists(DATA_PATH) else 0.0
    meta: dict = {}
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    meta["source_csv_mtime"] = mtime
    meta["dashboard_corpus_path"] = "dashboard_corpus.csv"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"metadata.json diperbarui (source_csv_mtime={mtime}).")


if __name__ == "__main__":
    main()
