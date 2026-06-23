import os
import re
import json
import warnings
from collections import Counter
from typing import Dict, List


import nltk

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from wordcloud import WordCloud

warnings.filterwarnings("ignore")


# =====================================================
# KONFIGURASI PATH
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "ulasan_honda_ecare_5k.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def read_dataset(path: str) -> pd.DataFrame:
    """
    Membaca dataset secara robust.
    Mendukung dua format:
    1) Memiliki kolom `content` dan `label`
    2) Memiliki kolom `content` dan `score` (label diturunkan dari skor)
    """
    try:
        # Prioritas format default CSV koma
        df = pd.read_csv(path, on_bad_lines="skip")
    except Exception:
        df = pd.DataFrame()

    if "content" not in df.columns:
        # Fallback untuk dataset delimiter titik-koma
        df = pd.read_csv(path, sep=";", engine="python", on_bad_lines="skip")

    # Bersihkan nama kolom yang kosong/berlebih
    df = df.loc[:, ~df.columns.astype(str).str.contains(r"^Unnamed", na=False)]
    df = df.loc[:, [col for col in df.columns if str(col).strip() != ""]]

    if "content" not in df.columns:
        raise ValueError("Kolom `content` tidak ditemukan pada dataset.")

    if "label" not in df.columns:
        if "score" not in df.columns:
            raise ValueError("Kolom `label` atau `score` tidak ditemukan pada dataset.")
        score_num = pd.to_numeric(df["score"], errors="coerce")
        # Aturan pelabelan otomatis:
        # score >= 4 -> positif, score <= 2 -> negatif, score 3 di-drop agar polaritas jelas.
        df["label"] = np.select(
            [score_num >= 4, score_num <= 2],
            ["positif", "negatif"],
            default=None,
        )
        df = df.dropna(subset=["label"])
    else:
        df["label"] = df["label"].astype(str).str.lower().str.strip()
        label_map = {
            "positive": "positif",
            "negative": "negatif",
            "pos": "positif",
            "neg": "negatif",
            "1": "positif",
            "0": "negatif",
        }
        df["label"] = df["label"].replace(label_map)
        df = df[df["label"].isin(["positif", "negatif"])]

    return df


# =====================================================
# RESOURCE PREPROCESSING
# =====================================================
def ensure_nltk_resources() -> None:
    """Memastikan resource NLTK tersedia."""
    import nltk

    resources = [
        ("corpora/stopwords", "stopwords"),
        ("tokenizers/punkt", "punkt"),
    ]
    for resource_path, resource_name in resources:
        try:
            nltk.data.find(resource_path)
        except (LookupError, OSError):
            try:
                nltk.download(resource_name, quiet=True)
            except Exception:
                try:
                    nltk.download("punkt", quiet=True)
                except Exception:
                    pass

    # Extra fallback for 'punkt_tab' if referenced by environment/code
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except Exception:
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            pass


ensure_nltk_resources()

INDO_STOPWORDS = set(stopwords.words("indonesian"))
STEMMER = StemmerFactory().create_stemmer()

NORMALIZATION_DICT = {
    "gk": "tidak",
    "ga": "tidak",
    "gak": "tidak",
    "nggak": "tidak",
    "ngga": "tidak",
    "tdk": "tidak",
    "bgt": "banget",
    "bgtt": "banget",
    "bangettt": "banget",
    "apk": "aplikasi",
    "app": "aplikasi",
    "udh": "sudah",
    "udah": "sudah",
    "blm": "belum",
    "dr": "dari",
    "krn": "karena",
    "karna": "karena",
    "dgn": "dengan",
    "sy": "saya",
    "aja": "saja",
    "tp": "tapi",
    "jg": "juga",
    "trs": "terus",
    "sm": "sama",
    "mantul": "mantap",
}


# =====================================================
# FUNGSI PREPROCESSING
# =====================================================
def remove_repeated_characters(text: str) -> str:
    """Mengubah karakter berulang menjadi karakter tunggal."""
    return re.sub(r"(.)\1{2,}", r"\1", text)


def clean_text(text: str) -> str:
    """Membersihkan teks dari noise."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#\w+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"[\U00010000-\U0010ffff]", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = remove_repeated_characters(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_tokens(tokens: List[str]) -> List[str]:
    """Normalisasi sederhana berdasarkan kamus slang."""
    return [NORMALIZATION_DICT.get(token, token) for token in tokens]


def preprocess_text(text: str) -> str:
    """Pipeline preprocessing lengkap."""
    text = str(text)
    text = clean_text(text)
    tokens = word_tokenize(text)
    tokens = normalize_tokens(tokens)
    tokens = [token for token in tokens if token not in INDO_STOPWORDS and len(token) > 1]
    tokens = [STEMMER.stem(token) for token in tokens]
    return " ".join(tokens)


# =====================================================
# UTILITAS VISUALISASI DAN EVALUASI
# =====================================================
def save_plot(filename: str) -> None:
    path = os.path.join(REPORTS_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def save_confusion_matrix_comparison(
    cm_nb: np.ndarray,
    cm_svm: np.ndarray,
    class_names: List[str],
    filename: str = "confusion_matrix_perbandingan.png",
) -> None:
    """Simpan confusion matrix NB dan SVM dalam satu gambar."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    matrices = [cm_nb, cm_svm]
    titles = ["Naive Bayes", "SVM (LinearSVC)"]
    cmaps = ["Blues", "Greens"]

    for ax, cm, title, cmap in zip(axes, matrices, titles, cmaps):
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap=cmap,
            cbar=False,
            ax=ax,
            xticklabels=class_names,
            yticklabels=class_names,
        )
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    fig.suptitle("Perbandingan Confusion Matrix: Naive Bayes vs SVM", fontsize=14, y=1.02)
    path = os.path.join(REPORTS_DIR, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_metrics_comparison(
    comparison_metrics: pd.DataFrame,
    filename: str = "perbandingan_metrik_lengkap.png",
) -> None:
    """Simpan perbandingan Accuracy, Precision, Recall, dan F1-score dalam satu gambar."""
    metric_cols = ["Accuracy", "Precision", "Recall", "F1-score"]
    long_df = comparison_metrics.melt(
        id_vars="Model",
        value_vars=metric_cols,
        var_name="Metrik",
        value_name="Nilai",
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=long_df, x="Metrik", y="Nilai", hue="Model", palette="Set2", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Skor")
    ax.set_title("Perbandingan Metrik Evaluasi: Naive Bayes vs SVM")
    ax.legend(title="Model", loc="lower right")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=2, fontsize=9)

    path = os.path.join(REPORTS_DIR, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_svm_prediction_distribution(
    labels: List[str],
    filename: str = "distribusi_prediksi_svm.png",
) -> None:
    """Simpan grafik distribusi sentimen hasil prediksi SVM pada keseluruhan dataset."""
    counts = pd.Series(labels).value_counts().reindex(["negatif", "positif"])
    percentages = (counts / counts.sum() * 100).round(2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.barplot(x=counts.index, y=counts.values, palette="viridis", ax=axes[0])
    axes[0].set_title("Distribusi Prediksi SVM (Jumlah)")
    axes[0].set_xlabel("Sentimen")
    axes[0].set_ylabel("Jumlah Ulasan")
    for idx, value in enumerate(counts.values):
        axes[0].text(idx, value + 8, f"{int(value)}", ha="center", fontsize=11)

    sns.barplot(x=percentages.index, y=percentages.values, palette="magma", ax=axes[1])
    axes[1].set_title("Distribusi Prediksi SVM (Persentase)")
    axes[1].set_xlabel("Sentimen")
    axes[1].set_ylabel("Persentase (%)")
    axes[1].set_ylim(0, 100)
    for idx, value in enumerate(percentages.values):
        axes[1].text(idx, value + 2, f"{value:.2f}%", ha="center", fontsize=11)

    fig.suptitle(
        "Distribusi Sentimen Hasil Prediksi Model SVM (Keseluruhan Dataset)",
        fontsize=14,
        y=1.02,
    )
    path = os.path.join(REPORTS_DIR, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def get_top_word_counts(texts: pd.Series, top_n: int = 10) -> pd.Series:
    """Hitung frekuensi kata dan ambil top-N."""
    counter: Counter = Counter()
    for text in texts.astype(str):
        counter.update(text.split())
    if not counter:
        return pd.Series(dtype=int)
    return pd.Series(dict(counter.most_common(top_n)))


def save_top_words_bar_chart(
    df: pd.DataFrame,
    top_n: int = 10,
    filename: str = "top10_kata_bar_chart.png",
) -> None:
    """Bar chart top kata paling sering: sentimen positif dan negatif."""
    top_pos = get_top_word_counts(df.loc[df["label"] == "positif", "cleaned_text"], top_n)
    top_neg = get_top_word_counts(df.loc[df["label"] == "negatif", "cleaned_text"], top_n)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    charts = [
        (top_pos, "Sentimen Positif", "Greens"),
        (top_neg, "Sentimen Negatif", "Reds"),
    ]

    for ax, (top_words, title, color) in zip(axes, charts):
        plot_df = top_words.sort_values().reset_index()
        plot_df.columns = ["Kata", "Frekuensi"]
        sns.barplot(data=plot_df, y="Kata", x="Frekuensi", palette=color, ax=ax)
        ax.set_title(f"Top {top_n} Kata — {title}")
        ax.set_xlabel("Frekuensi")
        ax.set_ylabel("Kata")
        for idx, value in enumerate(plot_df["Frekuensi"]):
            ax.text(value + 0.3, idx, f"{int(value)}", va="center", fontsize=9)

    fig.suptitle(f"Top {top_n} Kata Paling Sering Muncul (Positif vs Negatif)", fontsize=14, y=1.02)
    path = os.path.join(REPORTS_DIR, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_top_words_wordcloud(
    df: pd.DataFrame,
    top_n: int = 10,
    filename: str = "top10_kata_wordcloud.png",
) -> None:
    """Word cloud top kata paling sering: sentimen positif dan negatif."""
    top_pos = get_top_word_counts(df.loc[df["label"] == "positif", "cleaned_text"], top_n)
    top_neg = get_top_word_counts(df.loc[df["label"] == "negatif", "cleaned_text"], top_n)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    clouds = [
        (top_pos, "WordCloud Top 10 — Positif", "Greens"),
        (top_neg, "WordCloud Top 10 — Negatif", "Reds"),
    ]

    for ax, (top_words, title, cmap) in zip(axes, clouds):
        freq = top_words.to_dict() if not top_words.empty else {"kosong": 1}
        wc = WordCloud(
            width=900,
            height=450,
            background_color="white",
            colormap=cmap,
        ).generate_from_frequencies(freq)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(title)

    fig.suptitle("Word Cloud Top 10 Kata Paling Sering (Positif vs Negatif)", fontsize=14, y=1.02)
    path = os.path.join(REPORTS_DIR, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def evaluate_model(y_true, y_pred, model_name: str, y_train_true=None, y_train_pred=None) -> Dict[str, float]:
    """Menghitung metrik evaluasi model."""
    result = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-score": f1_score(y_true, y_pred),
    }
    if y_train_true is not None and y_train_pred is not None:
        result["Train Accuracy"] = accuracy_score(y_train_true, y_train_pred)
    return result


def main() -> None:
    sns.set(style="whitegrid")

    # =====================================================
    # 1. LOAD DATASET
    # =====================================================
    print("\n=== 1. LOAD DATASET ===")
    df = read_dataset(DATA_PATH)
    print("\n5 Data Teratas:")
    print(df.head())
    print("\nInfo Dataset:")
    print(df.info())

    total_before = len(df)
    print(f"\nJumlah Data Sebelum Cleaning: {total_before}")
    print("\nMissing Value:")
    print(df.isnull().sum())

    df = df.dropna(subset=["content", "label"]).reset_index(drop=True)

    before_dedup = len(df)
    df = df.drop_duplicates(subset=["content"]).reset_index(drop=True)
    total_after = len(df)
    print(f"Duplikat dihapus: {before_dedup - total_after}")
    print(f"Jumlah Data Sesudah Cleaning: {total_after}")

    # =====================================================
    # 2. EXPLORATORY DATA ANALYSIS (EDA)
    # =====================================================
    print("\n=== 2. EXPLORATORY DATA ANALYSIS ===")
    label_counts = df["label"].value_counts()
    label_percentages = (df["label"].value_counts(normalize=True) * 100).round(2)
    print("\nDistribusi Label:")
    print(label_counts)
    print("\nPersentase Label (%):")
    print(label_percentages)

    df["text_length"] = df["content"].astype(str).apply(len)
    print("\nStatistik Panjang Teks:")
    print(df["text_length"].describe())

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="label", palette="viridis")
    plt.title("Distribusi Label Sentimen")
    plt.xlabel("Label")
    plt.ylabel("Jumlah")
    save_plot("countplot_label.png")

    plt.figure(figsize=(10, 5))
    sns.histplot(df["text_length"], bins=30, kde=True, color="steelblue")
    plt.title("Distribusi Panjang Karakter Ulasan")
    plt.xlabel("Panjang Karakter")
    plt.ylabel("Frekuensi")
    save_plot("histogram_panjang_teks.png")

    # =====================================================
    # 3. TEXT PREPROCESSING
    # =====================================================
    print("\n=== 3. TEXT PREPROCESSING ===")
    print("\n5 Data Sebelum Preprocessing:")
    print(df[["content", "label"]].head())

    df["cleaned_text"] = df["content"].astype(str).apply(preprocess_text)
    df = df[df["cleaned_text"].str.strip() != ""].reset_index(drop=True)

    print("\n5 Data Sesudah Preprocessing:")
    print(df[["cleaned_text", "label"]].head())

    comparison_df = df[["content", "cleaned_text"]].head()
    print("\nPerbandingan Sebelum vs Sesudah:")
    print(comparison_df)

    # =====================================================
    # 4. WORDCLOUD
    # =====================================================
    print("\n=== 4. WORDCLOUD ===")
    positive_text = " ".join(df[df["label"] == "positif"]["cleaned_text"])
    negative_text = " ".join(df[df["label"] == "negatif"]["cleaned_text"])

    wc_positive = WordCloud(width=1000, height=500, background_color="white", colormap="Greens").generate(
        positive_text if positive_text else "positif"
    )
    wc_negative = WordCloud(width=1000, height=500, background_color="white", colormap="Reds").generate(
        negative_text if negative_text else "negatif"
    )

    plt.figure(figsize=(12, 6))
    plt.imshow(wc_positive, interpolation="bilinear")
    plt.axis("off")
    plt.title("WordCloud Sentimen Positif")
    save_plot("wordcloud_positif.png")

    plt.figure(figsize=(12, 6))
    plt.imshow(wc_negative, interpolation="bilinear")
    plt.axis("off")
    plt.title("WordCloud Sentimen Negatif")
    save_plot("wordcloud_negatif.png")

    print("\nTop 10 kata sentimen positif:")
    print(get_top_word_counts(df.loc[df["label"] == "positif", "cleaned_text"], 10))
    print("\nTop 10 kata sentimen negatif:")
    print(get_top_word_counts(df.loc[df["label"] == "negatif", "cleaned_text"], 10))
    save_top_words_bar_chart(df)
    save_top_words_wordcloud(df)

    # =====================================================
    # 5. LABEL ENCODING
    # =====================================================
    print("\n=== 5. LABEL ENCODING ===")
    le = LabelEncoder()
    le.fit(["negatif", "positif"])
    df["label_encoded"] = le.transform(df["label"])
    print(df[["label", "label_encoded"]].drop_duplicates().sort_values("label_encoded"))

    # =====================================================
    # 6. TF-IDF
    # =====================================================
    print("\n=== 6. TF-IDF ===")
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = tfidf.fit_transform(df["cleaned_text"])
    y = df["label_encoded"].values

    print(f"Shape TF-IDF: {X.shape}")
    feature_names = tfidf.get_feature_names_out()
    print("Contoh Fitur TF-IDF:")
    print(feature_names[:20])

    tfidf_means = np.asarray(X.mean(axis=0)).flatten()
    top_indices = tfidf_means.argsort()[-10:][::-1]
    top_words = [(feature_names[idx], tfidf_means[idx]) for idx in top_indices]
    print("10 Kata dengan Bobot TF-IDF Tertinggi:")
    for word, score in top_words:
        print(f"- {word}: {score:.6f}")

    # =====================================================
    # 7. SPLIT DATA
    # =====================================================
    print("\n=== 7. SPLIT DATA ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Jumlah Data Train: {X_train.shape[0]}")
    print(f"Jumlah Data Test: {X_test.shape[0]}")

    # =====================================================
    # 8. TRAINING MODEL
    # =====================================================
    print("\n=== 8. TRAINING MODEL ===")
    nb_model = MultinomialNB()
    svm_model = LinearSVC(class_weight="balanced", random_state=42)

    nb_model.fit(X_train, y_train)
    svm_model.fit(X_train, y_train)

    y_pred_nb = nb_model.predict(X_test)
    y_pred_svm = svm_model.predict(X_test)
    y_train_pred_nb = nb_model.predict(X_train)
    y_train_pred_svm = svm_model.predict(X_train)

    # =====================================================
    # 9. EVALUASI MODEL
    # =====================================================
    print("\n=== 9. EVALUASI MODEL ===")
    nb_metrics = evaluate_model(y_test, y_pred_nb, "Naive Bayes", y_train, y_train_pred_nb)
    svm_metrics = evaluate_model(y_test, y_pred_svm, "SVM (LinearSVC)", y_train, y_train_pred_svm)

    print("\nEvaluasi Naive Bayes:")
    print(f"Train Accuracy: {nb_metrics['Train Accuracy']:.4f}")
    print(f"Test Accuracy : {nb_metrics['Accuracy']:.4f}")
    print(f"Precision     : {nb_metrics['Precision']:.4f}")
    print(f"Recall        : {nb_metrics['Recall']:.4f}")
    print(f"F1-score      : {nb_metrics['F1-score']:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred_nb, target_names=le.classes_))

    print("\nEvaluasi SVM (LinearSVC):")
    print(f"Train Accuracy: {svm_metrics['Train Accuracy']:.4f}")
    print(f"Test Accuracy : {svm_metrics['Accuracy']:.4f}")
    print(f"Precision     : {svm_metrics['Precision']:.4f}")
    print(f"Recall        : {svm_metrics['Recall']:.4f}")
    print(f"F1-score      : {svm_metrics['F1-score']:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred_svm, target_names=le.classes_))

    cm_nb = confusion_matrix(y_test, y_pred_nb)
    cm_svm = confusion_matrix(y_test, y_pred_svm)

    # =====================================================
    # 10. VISUALISASI EVALUASI
    # =====================================================
    print("\n=== 10. VISUALISASI EVALUASI ===")
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_nb, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix Naive Bayes")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    save_plot("confusion_matrix_nb.png")

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_svm, annot=True, fmt="d", cmap="Greens", cbar=False)
    plt.title("Confusion Matrix SVM (LinearSVC)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    save_plot("confusion_matrix_svm.png")
    save_confusion_matrix_comparison(cm_nb, cm_svm, list(le.classes_))

    comparison_metrics = pd.DataFrame([nb_metrics, svm_metrics])[
        ["Model", "Accuracy", "Precision", "Recall", "F1-score"]
    ]

    save_metrics_comparison(comparison_metrics)

    y_pred_all_svm = le.inverse_transform(svm_model.predict(X))
    save_svm_prediction_distribution(list(y_pred_all_svm))
    print("\nDistribusi prediksi SVM (keseluruhan dataset):")
    print(pd.Series(y_pred_all_svm).value_counts())
    print((pd.Series(y_pred_all_svm).value_counts(normalize=True) * 100).round(2))

    metric_names = ["Accuracy", "Precision", "Recall", "F1-score"]
    for metric in metric_names:
        plt.figure(figsize=(7, 4))
        sns.barplot(data=comparison_metrics, x="Model", y=metric, palette="Set2")
        plt.ylim(0, 1)
        plt.title(f"Perbandingan {metric} NB vs SVM")
        plt.ylabel(metric)
        save_plot(f"perbandingan_{metric.lower().replace('-', '_')}.png")

    # =====================================================
    # 11. CROSS VALIDATION
    # =====================================================
    print("\n=== 11. CROSS VALIDATION (5-FOLD) ===")
    nb_cv_scores = cross_val_score(MultinomialNB(), X, y, cv=5, scoring="accuracy")
    svm_calibrated = CalibratedClassifierCV(
        estimator=LinearSVC(class_weight="balanced", random_state=42), cv=3
    )
    svm_cv_scores = cross_val_score(svm_calibrated, X, y, cv=5, scoring="accuracy")

    print(f"Naive Bayes CV Accuracy Mean: {nb_cv_scores.mean():.4f}")
    print(f"SVM CV Accuracy Mean        : {svm_cv_scores.mean():.4f}")

    # =====================================================
    # 12. PERBANDINGAN MODEL
    # =====================================================
    print("\n=== 12. PERBANDINGAN MODEL ===")
    print(comparison_metrics)

    best_by_accuracy = comparison_metrics.loc[comparison_metrics["Accuracy"].idxmax(), "Model"]
    best_by_f1 = comparison_metrics.loc[comparison_metrics["F1-score"].idxmax(), "Model"]
    conclusion = (
        f"Model terbaik berdasarkan akurasi adalah {best_by_accuracy}, "
        f"dan berdasarkan F1-score adalah {best_by_f1}."
    )
    print("Kesimpulan Otomatis:")
    print(conclusion)

    # =====================================================
    # 13. PREDIKSI KALIMAT BARU
    # =====================================================
    print("\n=== 13. PREDIKSI KALIMAT BARU ===")
    contoh_text = [
        "Aplikasi sangat membantu dan mudah digunakan",
        "Aplikasi error dan sangat mengecewakan",
    ]
    cleaned_examples = [preprocess_text(text) for text in contoh_text]
    vectorized_examples = tfidf.transform(cleaned_examples)
    example_predictions = svm_model.predict(vectorized_examples)
    example_labels = le.inverse_transform(example_predictions)

    for original, cleaned, pred in zip(contoh_text, cleaned_examples, example_labels):
        print(f"Teks Asli       : {original}")
        print(f"Hasil Cleaning  : {cleaned}")
        print(f"Prediksi        : {pred}")
        print("-" * 50)

    # =====================================================
    # 14. SAVE MODEL
    # =====================================================
    print("\n=== 14. SAVE MODEL ===")
    joblib.dump(nb_model, os.path.join(MODELS_DIR, "nb_model.pkl"))
    joblib.dump(svm_model, os.path.join(MODELS_DIR, "svm_model.pkl"))
    joblib.dump(tfidf, os.path.join(MODELS_DIR, "tfidf.pkl"))
    joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.pkl"))

    # Menyimpan artefak tambahan untuk dashboard
    preprocessor_bundle = {
        "normalization_dict": NORMALIZATION_DICT,
        "stopwords": sorted(list(INDO_STOPWORDS)),
    }
    with open(os.path.join(MODELS_DIR, "preprocessor_config.json"), "w", encoding="utf-8") as f:
        json.dump(preprocessor_bundle, f, ensure_ascii=False, indent=2)

    comparison_metrics.to_csv(os.path.join(MODELS_DIR, "evaluation_metrics.csv"), index=False)
    cv_results = pd.DataFrame(
        {
            "Model": ["Naive Bayes", "SVM (LinearSVC)"],
            "CV Mean Accuracy": [nb_cv_scores.mean(), svm_cv_scores.mean()],
            "CV Std": [nb_cv_scores.std(), svm_cv_scores.std()],
        }
    )
    cv_results.to_csv(os.path.join(MODELS_DIR, "cv_results.csv"), index=False)

    source_csv_mtime = os.path.getmtime(DATA_PATH) if os.path.exists(DATA_PATH) else 0.0
    corpus_path = os.path.join(MODELS_DIR, "dashboard_corpus.csv")
    df[["content", "label", "cleaned_text"]].to_csv(
        corpus_path, index=False, encoding="utf-8"
    )

    metadata = {
        "total_data_before_cleaning": int(total_before),
        "total_data_after_cleaning": int(total_after),
        "train_size": int(X_train.shape[0]),
        "test_size": int(X_test.shape[0]),
        "best_model_by_accuracy": best_by_accuracy,
        "best_model_by_f1": best_by_f1,
        "conclusion": conclusion,
        "source_csv_mtime": source_csv_mtime,
        "dashboard_corpus_path": "dashboard_corpus.csv",
    }
    with open(os.path.join(MODELS_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("Model dan artefak evaluasi berhasil disimpan ke folder models/ dan reports/.")


if __name__ == "__main__":
    main()
