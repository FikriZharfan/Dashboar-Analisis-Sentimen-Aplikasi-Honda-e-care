"""
Dashboard Analisis Sentimen — Honda e-Care
Tema: modern automotive analytics · abu metalik · aksen merah Honda
"""

import html as html_module
import json
import os
import re
import time
import unicodedata
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud

# =====================================================
# KONFIGURASI DASAR
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "ulasan_honda_ecare_5k.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
DASHBOARD_CORPUS_PATH = os.path.join(MODELS_DIR, "dashboard_corpus.csv")
METADATA_PATH = os.path.join(MODELS_DIR, "metadata.json")

# Palet Honda Analytics
C_DARK = "#2B2B2B"
C_METAL = "#4F4F4F"
C_LIGHT = "#E5E5E5"
C_WHITE = "#FFFFFF"
C_ACCENT = "#D50000"
C_POSITIVE = "#2E7D32"
C_NEGATIVE = "#C62828"

MENU_OPTIONS = [
    "Halaman Utama",
    "Visualisasi",
    "Prediksi Sentimen",
    "Hasil Evaluasi",
]
NAV_LABELS = [
    "🏠  Halaman Utama",
    "📊  Visualisasi",
    "💬  Prediksi Sentimen",
    "📋  Hasil Evaluasi",
]


PLOTLY_LAYOUT = dict(
    paper_bgcolor=C_DARK,
    plot_bgcolor=C_DARK,
    font=dict(family="Inter, Poppins, sans-serif", color="#E8E8E8", size=13),
    margin=dict(l=58, r=130, t=58, b=64),
    colorway=[C_ACCENT, C_METAL, C_LIGHT, "#9E9E9E"],
    xaxis=dict(
        gridcolor="#404040",
        zerolinecolor="#505050",
        tickfont=dict(color="#E8E8E8"),
        title=dict(font=dict(color="#E8E8E8")),
        linecolor="#666666",
    ),
    yaxis=dict(
        gridcolor="#404040",
        zerolinecolor="#505050",
        tickfont=dict(color="#E8E8E8"),
        title=dict(font=dict(color="#E8E8E8")),
        linecolor="#666666",
    ),
    legend=dict(
        bgcolor=C_DARK,
        bordercolor="#5A5A5A",
        borderwidth=1,
        font=dict(color="#E8E8E8", size=12),
    ),
)

st.set_page_config(
    page_title="Honda e-Care · Sentiment Analytics",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =====================================================
# TEMA & CSS
# =====================================================
def inject_custom_css() -> None:
    """Menyuntikkan stylesheet global — nuansa dashboard otomotif premium."""
    # Hindari f-string agar karakter '{' '}' di CSS tidak memicu SyntaxError.
    css = """

<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Poppins:wght@400;500;600;700&display=swap" rel="stylesheet">

<style>
:root {
    --honda-dark: __C_DARK__;
    --honda-metal: __C_METAL__;
    --honda-light: __C_LIGHT__;
    --honda-white: __C_WHITE__;
    --honda-red: __C_ACCENT__;
}

html, body, [class*="css"] {
    font-family: 'Inter', 'Poppins', sans-serif !important;
}

.stApp {
    background: linear-gradient(160deg, #CFCFCF 0%, #E2E2E2 45%, #F2F2F2 100%);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, __C_DARK__ 0%, #1F1F1F 100%) !important;
    border-right: 1px solid #3A3A3A;
}
[data-testid="stSidebar"] * {
    color: __C_LIGHT__ !important;
}

[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stFileUploader label,
[data-testid="stSidebar"] .honda-nav-label {
    color: #BDBDBD !important;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* Sembunyikan menu bawaan Streamlit — JANGAN sembunyikan header (tombol sidebar) */
#MainMenu, footer {
    visibility: hidden;
    height: 0;
}
header[data-testid="stHeader"] {
    visibility: visible !important;
    background: rgba(255,255,255,0) !important;
}

[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"],
button[data-testid="stSidebarCollapsedControl"],
button[data-testid="baseButton-header"] {
    visibility: visible !important;
    display: flex !important;
    color: __C_DARK__ !important;
}

/* Navigasi sidebar — st.radio native */
[data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid transparent !important;
    border-radius: 8px !important;
    padding: 0.42rem 0.65rem !important;
    margin: 0.2rem 0 !important;
    width: 100%;
    transition: background 0.2s, border-color 0.2s;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:hover {
    background: rgba(255,255,255,0.12) !important;
    border-color: #5A5A5A !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child {
    display: none !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"] p {
    color: __C_LIGHT__ !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
    background: __C_ACCENT__ !important;
    border-color: __C_ACCENT__ !important;
    margin: 0.2rem 0 !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) p {
    color: __C_WHITE__ !important;
    font-weight: 600 !important;
}

.honda-sidebar-brand {
    padding: 1.1rem 1rem 0.9rem 1rem;
    margin-bottom: 0.75rem;
    background: linear-gradient(180deg, rgba(213,0,0,0.95), rgba(229,57,53,0.9));
    border-radius: 18px;
    box-shadow: 0 16px 40px rgba(0, 0, 0, 0.24);
}

.honda-sidebar-title {
    color: __C_WHITE__;
    font-size: 1.6rem;
    font-weight: 800;
    letter-spacing: 0.22em;
    margin-bottom: 0.35rem;
}

.honda-sidebar-subtitle {
    color: rgba(255, 255, 255, 0.82);
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.85rem;
}

.honda-sidebar-divider {
    width: 52px;
    height: 4px;
    background: rgba(255, 255, 255, 0.95);
    border-radius: 999px;
}

.honda-sidebar-section {
    color: #BDBDBD !important;
    font-size: 0.76rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin: 0.5rem 0 0.35rem 0;
    padding-left: 0.1rem;
}

.honda-sidebar-spacer {
    height: 0.4rem;
}

.honda-sidebar-meta {
    margin-top: 1rem;
    padding: 0.95rem 0.95rem 1rem 0.95rem;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    background: rgba(255, 255, 255, 0.02);
}

.honda-sidebar-meta-row {
    color: rgba(255, 255, 255, 0.82);
    font-size: 0.84rem;
    line-height: 1.7;
    margin-bottom: 0.35rem;
}

.honda-sidebar-meta-row:last-child {
    margin-bottom: 0;
}

.honda-header {
    background: linear-gradient(135deg, __C_DARK__ 0%, __C_METAL__ 55%, #3A3A3A 100%);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 12px 40px rgba(43,43,43,0.25);
    border-left: 5px solid __C_ACCENT__;
    animation: fadeSlideDown 0.55s ease-out;
}

.honda-header h1 {
    color: __C_WHITE__;
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    font-size: 1.85rem;
    margin: 0 0 0.35rem 0;
    letter-spacing: -0.02em;
}

.honda-header .subtitle {
    color: #B0B0B0;
    font-size: 0.95rem;
    margin: 0;
    line-height: 1.5;
}

.honda-header .badge {
    display: inline-block;
    background: __C_ACCENT__;
    color: white;
    font-size: 0.68rem;
    font-weight: 600;
    padding: 0.25rem 0.65rem;
    border-radius: 4px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
}

.section-title {
    font-family: 'Poppins', sans-serif;
    font-weight: 600;
    color: __C_DARK__;
    font-size: 1.05rem;
    margin: 1.1rem 0 0.45rem 0;
    padding-bottom: 0.35rem;
    border-bottom: 2px solid __C_ACCENT__;
    display: inline-block;
    animation: fadeIn 0.4s ease;
}

.st-key-honda_home_chart,
.st-key-honda_home_table {
    padding: 0.25rem 0.35rem 0.4rem !important;
}

[data-testid="stVerticalBlockBorderWrapper"]:has(.st-key-honda_home_chart),
[data-testid="stVerticalBlockBorderWrapper"]:has(.st-key-honda_home_table) {
    background: __C_WHITE__ !important;
    border-color: #C8C8C8 !important;
    box-shadow: 0 4px 18px rgba(43,43,43,0.06) !important;
}

.honda-insight-card {
    background: linear-gradient(152deg, __C_DARK__ 0%, #353535 48%, #2F2F2F 100%);
    border-radius: 14px;
    border: 1px solid #4A4A4A;
    border-left: 4px solid __C_ACCENT__;
    padding: 1.35rem 1.5rem 1.25rem;
    box-shadow: 0 10px 32px rgba(0,0,0,0.2);
    margin: 0.35rem 0 1.15rem 0;
}
.honda-insight-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.95rem;
    padding-bottom: 0.1rem;
}
.honda-insight-icon {
    font-size: 1.45rem;
    line-height: 1;
    flex-shrink: 0;
    filter: drop-shadow(0 1px 2px rgba(0,0,0,0.3));
}
.honda-insight-title {
    margin: 0;
    font-family: 'Poppins', sans-serif;
    font-size: 1.08rem;
    font-weight: 600;
    color: __C_WHITE__;
    letter-spacing: 0.015em;
    line-height: 1.3;
}
.honda-insight-body {
    margin: 0 0 1.15rem 0;
    padding-right: 0.25rem;
    color: #C4C4C4;
    font-size: 0.94rem;
    line-height: 1.7;
    font-weight: 400;
}
.honda-insight-bar-outer {
    margin-top: 0.15rem;
    padding: 0 0.1rem;
}
.honda-insight-bar-track {
    height: 7px;
    background: rgba(255,255,255,0.14);
    border-radius: 999px;
    overflow: hidden;
    margin: 0;
    box-shadow: inset 0 1px 2px rgba(0,0,0,0.25);
}
.honda-insight-bar-fill {
    height: 100%;
    min-width: 4px;
    background: linear-gradient(90deg, __C_ACCENT__ 0%, #FF1744 100%);
}
.honda-prediction-card {
    background: __C_WHITE__;
    border: 1px solid #BDBDBD;
    border-radius: 12px;
    padding: 1.15rem 1.3rem;
    margin: 0.85rem 0 1.15rem 0;
    box-shadow: 0 4px 16px rgba(43,43,43,0.08);
}

.sentiment-pos {
    background: linear-gradient(135deg, #E8F5E9, #C8E6C9);
    color: __C_POSITIVE__;
    border: 1px solid #A5D6A7;
}
.sentiment-neg {
    background: linear-gradient(135deg, #FFEBEE, #FFCDD2);
    color: __C_NEGATIVE__;
    border: 1px solid #EF9A9A;
}

.honda-footer {
    margin-top: 2.5rem;
    padding: 1.25rem 1.5rem;
    background: linear-gradient(90deg, __C_DARK__, __C_METAL__);
    border-radius: 12px;
    color: #BDBDBD;
    font-size: 0.8rem;
    text-align: center;
    box-shadow: 0 4px 16px rgba(0,0,0,0.12);
}

.honda-footer strong {
    color: __C_WHITE__;
}

.honda-footer .accent {
    color: __C_ACCENT__;
    font-weight: 600;
}   

/* =====================================================
   METRIC CARDS (Total dataset / Positif / Negatif / Model Terbaik)
   ===================================================== */
.metric-card {
    background: linear-gradient(152deg, #2B2B2B 0%, #343434 48%, #2F2F2F 100%);
    border: 1px solid #4A4A4A;
    border-left: 4px solid __C_METAL__;
    border-radius: 14px;
    padding: 0.95rem 1.05rem;
    box-shadow: 0 10px 32px rgba(0,0,0,0.18);
    margin: 0.35rem 0;
}
.metric-card.accent {
    border-left: 4px solid __C_ACCENT__;
}
.metric-card:hover,
.honda-insight-card:hover {
    transform: translateY(-3px);
    transition: all 0.25s ease;
    box-shadow: 0 14px 32px rgba(0,0,0,0.28);
}
.metric-card,
.honda-insight-card,
.honda-header {
    animation: fadeIn 0.45s ease;
}
.metric-card .label {
    color: #BDBDBD;
    font-family: 'Poppins', sans-serif;
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 0.02em;
    margin-bottom: 0.35rem;
}
.metric-card .value {
    color: __C_WHITE__;
    font-family: 'Poppins', sans-serif;
    font-weight: 800;
    font-size: 1.35rem;
    line-height: 1.2;
}
.metric-card.accent .value {
    color: __C_WHITE__;
}
.metric-card .hint {
    margin-top: 0.45rem;
    color: #C4C4C4;
    font-size: 0.82rem;
    line-height: 1.4;
}

/* Pastikan Streamlit container tidak menimpa warna card */
/* Selector diperkuat karena Streamlit kadang menimpa color pada wrapper */
.metric-card * {
    color: inherit;
}
.metric-card .label,
.metric-card .value,
.metric-card .hint,
.metric-card p,
.metric-card span,
.metric-card div {
    color: inherit;
}
.metric-card .label {
    color: #BDBDBD !important;
}
.metric-card .value,
.metric-card .value * {
    color: #FFFFFF !important;
    opacity: 1 !important;
}
.metric-card .hint {
    color: #C4C4C4 !important;
}
.metric-card.accent .value {
    color: __C_WHITE__ !important;
}

div.stButton > button {

    background: linear-gradient(135deg, __C_ACCENT__ 0%, #B00000 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.75rem !important;
    letter-spacing: 0.03em;
    transition: transform 0.2s, box-shadow 0.2s !important;
    box-shadow: 0 4px 14px rgba(213,0,0,0.35) !important;
}

div.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 22px rgba(213,0,0,0.45) !important;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeSlideDown {
    from { opacity: 0; transform: translateY(-12px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulse {
    0%, 100% { transform: scale(1); }
    50%      { transform: scale(1.02); }
}
</style>
"""

    css = (
        css.replace("__C_DARK__", C_DARK)
        .replace("__C_METAL__", C_METAL)
        .replace("__C_LIGHT__", C_LIGHT)
        .replace("__C_WHITE__", C_WHITE)
        .replace("__C_ACCENT__", C_ACCENT)
        .replace("__C_POSITIVE__", C_POSITIVE)
        .replace("__C_NEGATIVE__", C_NEGATIVE)
    )

    st.markdown(css, unsafe_allow_html=True)



def render_header() -> None:
    """Header premium bertema automotive analytics."""
    st.markdown(
        """
        <div class="honda-header">
            <span class="badge">Honda Analytics</span>
            <h1>Analisis Sentimen Ulasan Honda e-Care</h1>
            <p class="subtitle">
                Platform intelijen sentimen berbasis machine learning — Naive Bayes &amp; SVM (LinearSVC)
                · Dataset ulasan aplikasi · Visualisasi premium untuk penelitian &amp; portfolio.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.markdown(
        f"""
        <div class="honda-footer">
            <strong>Honda e-Care Sentiment Analytics Dashboard</strong><br>
            Powered by <span class="accent">Streamlit</span> · scikit-learn · Plotly
            &nbsp;|&nbsp; Tema abu metalik · Aksen <span class="accent">#D50000</span>
            &nbsp;|&nbsp; Portfolio Data Science
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_insight_card(
    title: str,
    content: str,
    *,
    bar_pct: int = 88,
    emoji: str = "📊",
) -> None:
    """
    Kartu kesimpulan/rekomendasi — ganti info_card hydralit (ikon FontAwesome sering tidak render).
    """
    pct = max(0, min(100, int(bar_pct)))
    safe_title = html_module.escape(title)
    safe_content = html_module.escape(content).replace("\n", "<br/>")
    st.markdown(
        f"""
        <div class="honda-insight-card">
            <div class="honda-insight-header">
                <span class="honda-insight-icon" aria-hidden="true">{emoji}</span>
                <h3 class="honda-insight-title">{safe_title}</h3>
            </div>
            <p class="honda-insight-body">{safe_content}</p>
            <div class="honda-insight-bar-outer">
                <div
                    class="honda-insight-bar-track"
                    role="progressbar"
                    aria-valuenow="{pct}"
                    aria-valuemin="0"
                    aria-valuemax="100"
                >
                    <div class="honda-insight-bar-fill" style="width: {pct}%;"></div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_result(cleaned: str, pred_label: str, model_name: str) -> None:
    """Menampilkan preprocessing + prediksi dengan kontras tinggi (bukan st.markdown default)."""
    safe_text = html_module.escape(cleaned)
    safe_model = html_module.escape(model_name)
    val_cls = "honda-prediction-value--pos" if pred_label == "positif" else "honda-prediction-value--neg"
    st.markdown(
        f"""
        <div class="honda-prediction-card">
            <div class="honda-prediction-row">
                <span class="honda-prediction-field-label">Hasil preprocessing</span>
                <code class="honda-prediction-code">{safe_text}</code>
            </div>
            <div class="honda-prediction-row honda-prediction-row--result">
                <span class="honda-prediction-field-label">Hasil prediksi</span>
                <span class="honda-prediction-value {val_cls}">{pred_label.upper()} · {safe_model}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card_html(label: str, value: str, hint: str = "", accent: bool = False) -> str:
    cls = "metric-card accent" if accent else "metric-card"
    hint_html = f'<div class="hint">{hint}</div>' if hint else ""
    return f"""
    <div class="{cls}">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        {hint_html}
    </div>
    """


def render_metric_row(cards: List[Tuple[str, str, str, bool]]) -> None:
    cols = st.columns(len(cards))
    for col, (label, value, hint, accent) in zip(cols, cards):
        with col:
            st.markdown(metric_card_html(label, value, hint, accent), unsafe_allow_html=True)


def section_title(text: str) -> None:
    st.markdown(f'<p class="section-title">{text}</p>', unsafe_allow_html=True)


def apply_plotly_theme(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(
            text=title,
            font=dict(size=16, color=C_WHITE),
            x=0.02,
            xanchor="left",
        ),
    )
    return fig


def show_plotly_chart(fig: go.Figure, *, key: Optional[str] = None) -> None:
    """Plotly figure tidak memakai theme Streamlit (hindari teks sumbu hilang di kartu terang)."""
    kw: Dict[str, object] = {"use_container_width": True, "theme": None}
    if key is not None:
        kw["key"] = key
    st.plotly_chart(fig, **kw)


# =====================================================
# NLTK & PREPROCESSING
# =====================================================
def ensure_nltk_resources() -> None:
    resources = ["stopwords", "punkt"]
    for resource in resources:
        try:
            if resource == "stopwords":
                nltk.data.find("corpora/stopwords")
            else:
                nltk.data.find("tokenizers/punkt")
        except (LookupError, OSError):
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                # best-effort: try explicit punkt download if the generic name failed
                try:
                    nltk.download("punkt", quiet=True)
                except Exception:
                    pass

    # Extra fallback: some environments or builds may reference 'punkt_tab'.
    # Attempt to ensure it's available (harmless if unavailable).
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except Exception:
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            pass


ensure_nltk_resources()
STEMMER = StemmerFactory().create_stemmer()


def load_preprocessor_config() -> Dict:
    config_path = os.path.join(MODELS_DIR, "preprocessor_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "normalization_dict": {
            "gk": "tidak", "ga": "tidak", "gak": "tidak",
            "nggak": "tidak", "tdk": "tidak", "bgt": "banget",
            "apk": "aplikasi", "app": "aplikasi",
        },
        "protected_stopwords": ["tidak", "bukan", "jangan", "belum", "kurang"],
        "negation_words": ["tidak", "bukan", "belum"],
        "negated_negative_terms": [
            "bug", "crash", "error", "eror", "forceclose", "hang",
            "lag", "lambat", "lemot", "macet", "ngelag", "ngefreeze", "susah",
        ],
        "negation_replacement_tokens": ["lancar"],
        "stopwords": stopwords.words("indonesian"),
    }


CONFIG = load_preprocessor_config()
NORMALIZATION_DICT = CONFIG["normalization_dict"]
PROTECTED_STOPWORDS = set(CONFIG.get("protected_stopwords", ["tidak", "bukan", "jangan", "belum", "kurang"]))
INDO_STOPWORDS = set(CONFIG["stopwords"]) - PROTECTED_STOPWORDS
NEGATION_WORDS = set(CONFIG.get("negation_words", ["tidak", "bukan", "belum"]))
NEGATED_NEGATIVE_TERMS = set(CONFIG.get("negated_negative_terms", [
    "bug", "crash", "error", "eror", "forceclose", "hang",
    "lag", "lambat", "lemot", "macet", "ngelag", "ngefreeze", "susah",
]))
NEGATION_REPLACEMENT_TOKENS = CONFIG.get("negation_replacement_tokens", ["lancar"])


def read_dataset(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, on_bad_lines="skip")
    except Exception:
        df = pd.DataFrame()
    if "content" not in df.columns:
        df = pd.read_csv(path, sep=";", engine="python", on_bad_lines="skip")
    df = df.loc[:, ~df.columns.astype(str).str.contains(r"^Unnamed", na=False)]
    df = df.loc[:, [col for col in df.columns if str(col).strip() != ""]]
    if "content" not in df.columns:
        raise ValueError("Kolom `content` tidak ditemukan pada dataset.")
    if "label" not in df.columns and "score" in df.columns:
        score_num = pd.to_numeric(df["score"], errors="coerce")
        df["label"] = pd.Series(pd.NA, index=df.index, dtype="object")
        df.loc[score_num >= 4, "label"] = "positif"
        df.loc[score_num <= 2, "label"] = "negatif"
        df = df.dropna(subset=["label"])
    elif "label" in df.columns:
        df["label"] = df["label"].astype(str).str.lower().str.strip()
        df["label"] = df["label"].replace({
            "positive": "positif", "negative": "negatif",
            "pos": "positif", "neg": "negatif",
            "1": "positif", "0": "negatif",
        })
        df = df[df["label"].isin(["positif", "negatif"])]
    else:
        raise ValueError("Kolom `label` atau `score` tidak ditemukan pada dataset.")
    return df


@st.cache_data
def load_dataset() -> pd.DataFrame:
    return (
        read_dataset(DATA_PATH)
        .dropna(subset=["content", "label"])
        .reset_index(drop=True)
    )


@st.cache_resource
def load_models(_artifact_signature: Tuple[float, ...] = (0.0,)):
    return (
        joblib.load(os.path.join(MODELS_DIR, "nb_model.pkl")),
        joblib.load(os.path.join(MODELS_DIR, "svm_model.pkl")),
        joblib.load(os.path.join(MODELS_DIR, "tfidf.pkl")),
        joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl")),
    )


def remove_repeated_characters(text: str) -> str:
    return re.sub(r"(.)\1{2,}", r"\1", text)


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#\w+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"[\U00010000-\U0010ffff]", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = remove_repeated_characters(text)
    return re.sub(r"\s+", " ", text).strip()


def preprocess_text(text: str) -> str:
    text = clean_text(text)
    tokens = word_tokenize(text)
    tokens = [NORMALIZATION_DICT.get(t, t) for t in tokens]
    tokens = apply_negation_rules(tokens)
    tokens = [t for t in tokens if t not in INDO_STOPWORDS and len(t) > 1]
    tokens = [STEMMER.stem(t) for t in tokens]
    return " ".join(tokens)


def apply_negation_rules(tokens: List[str]) -> List[str]:
    result: List[str] = []
    idx = 0
    while idx < len(tokens):
        current = tokens[idx]
        next_token = tokens[idx + 1] if idx + 1 < len(tokens) else ""
        if current in NEGATION_WORDS and next_token in NEGATED_NEGATIVE_TERMS:
            result.extend(NEGATION_REPLACEMENT_TOKENS)
            idx += 2
            continue
        result.append(current)
        idx += 1
    return result


def has_positive_negation(text: str) -> bool:
    cleaned = clean_text(text)
    if not cleaned:
        return False

    tokens = [NORMALIZATION_DICT.get(token, token) for token in word_tokenize(cleaned)]
    if len(tokens) < 2:
        return False

    for index, token in enumerate(tokens):
        if token not in NEGATION_WORDS:
            continue

        for offset in range(1, min(4, len(tokens) - index)):
            if tokens[index + offset] in NEGATED_NEGATIVE_TERMS:
                return True

    return False


def predict_sentiment(text: str, cleaned: str, model, tfidf, label_encoder) -> str:
    vec = tfidf.transform([cleaned])
    return str(label_encoder.inverse_transform(model.predict(vec))[0])


def _finalize_labeled_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=["content", "label"]).reset_index(drop=True)


def _try_load_training_corpus_snapshot(raw_csv_mtime: float) -> Optional[pd.DataFrame]:
    if not os.path.isfile(DASHBOARD_CORPUS_PATH):
        return None
    
    snap = pd.read_csv(DASHBOARD_CORPUS_PATH, on_bad_lines="skip")
    if not {"content", "label", "cleaned_text"}.issubset(snap.columns):
        return None
        
    return _finalize_labeled_df(snap)


@st.cache_data
def load_dataset_with_cleaned_text(_csv_mtime: float) -> pd.DataFrame:
    snap = _try_load_training_corpus_snapshot(_csv_mtime)
    if snap is not None:
        return snap
    df = load_dataset().copy()
    df = df.drop_duplicates(subset=["content"]).reset_index(drop=True)
    df["cleaned_text"] = df["content"].astype(str).apply(preprocess_text)
    df = df[df["cleaned_text"].str.strip() != ""].reset_index(drop=True)
    return df


# =====================================================
# VISUALISASI (PLOTLY & MATPLOTLIB)
# =====================================================
def plot_sentiment_distribution(
    df: pd.DataFrame,
    title: str = "Distribusi Sentimen",
    *,
    chart_height: Optional[int] = None,
) -> go.Figure:
    counts = df["label"].value_counts().reset_index()
    counts.columns = ["label", "count"]
    color_map = {"positif": C_POSITIVE, "negatif": C_NEGATIVE}
    fig = px.bar(
        counts, x="label", y="count", text="count",
        color="label", color_discrete_map=color_map,
        labels={"label": "Sentimen", "count": "Jumlah"},
    )
    fig.update_traces(
        textposition="outside",
        textfont=dict(color=C_WHITE, size=14),
        marker_line_color=C_DARK,
        marker_line_width=1,
    )
    fig = apply_plotly_theme(fig, title)
    if chart_height is not None:
        fig.update_layout(
            height=chart_height,
            margin=dict(l=48, r=104, t=52, b=60),
        )
    return fig


def plot_confusion_matrix(cm, class_names: List[str], title: str) -> go.Figure:
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=class_names, y=class_names,
        colorscale=[[0, "#3D3D3D"], [0.5, C_METAL], [1, C_ACCENT]],
        text=cm, texttemplate="%{text}", textfont=dict(color=C_WHITE, size=14),
        hoverongaps=False,
    ))
    fig = apply_plotly_theme(fig, title)
    fig.update_layout(
        xaxis=dict(
            title=dict(text="Predicted", font=dict(color="#E8E8E8")),
            tickfont=dict(color="#E8E8E8"),
        ),
        yaxis=dict(
            title=dict(text="Actual", font=dict(color="#E8E8E8")),
            tickfont=dict(color="#E8E8E8"),
            autorange="reversed",
        ),
    )
    return fig


def plot_metrics_comparison(metrics_df: pd.DataFrame) -> go.Figure:
    melted = metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig = px.bar(
        melted, x="Metric", y="Score", color="Model", barmode="group",
        color_discrete_sequence=[C_METAL, C_ACCENT],
    )
    fig.update_layout(yaxis=dict(range=[0, 1.05], tickformat=".0%"))
    fig.update_traces(marker_line_color=C_DARK, marker_line_width=0.5)
    return apply_plotly_theme(fig, "Perbandingan Metrik Model")


def plot_cv_results(cv_df: pd.DataFrame) -> go.Figure:
    colors = [C_ACCENT if "SVM" in m else C_METAL for m in cv_df["Model"]]
    fig = go.Figure(data=go.Bar(
        x=cv_df["Model"],
        y=cv_df["CV Mean Accuracy"],
        error_y=dict(type="data", array=cv_df["CV Std"].tolist()),
        marker_color=colors,
        text=[f"{v:.1%}" for v in cv_df["CV Mean Accuracy"]],
        textposition="outside",
    ))
    fig.update_layout(yaxis=dict(range=[0, 1.05], tickformat=".0%"), showlegend=False)
    return apply_plotly_theme(fig, "Cross-Validation (5-Fold)")


def generate_wordcloud(text: str, title: str, colormap: str):
    fig, ax = plt.subplots(figsize=(10, 3.8), facecolor=C_DARK)
    ax.set_facecolor(C_DARK)
    wc = WordCloud(
        width=1000, height=420,
        background_color=C_DARK,
        colormap=colormap,
        max_words=120,
        prefer_horizontal=0.85,
    ).generate(text if text else "data")
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, color=C_LIGHT, fontsize=13, pad=12, fontfamily="Poppins")
    return fig


def styled_dataframe(df: pd.DataFrame, height: int = 360) -> None:
    st.dataframe(
        df, use_container_width=True, hide_index=True, height=height,
        column_config={
            "content": st.column_config.TextColumn("Ulasan", width="large"),
            "label": st.column_config.TextColumn("Label", width="small"),
            "cleaned_text": st.column_config.TextColumn("Teks Bersih", width="medium"),
            "prediction": st.column_config.TextColumn("Prediksi", width="small"),
            "predicted_label": st.column_config.TextColumn("Prediksi", width="small"),
        },
    )


# =====================================================
# HALAMAN DASHBOARD
# =====================================================
def page_home(raw_df: pd.DataFrame, df: pd.DataFrame, metadata: Dict) -> None:
    n_pos = int((raw_df["label"] == "positif").sum())
    n_neg = int((raw_df["label"] == "negatif").sum())
    best = metadata.get("best_model_by_accuracy", "SVM (LinearSVC)") if metadata else "—"
    pct_pos = (n_pos / len(raw_df) * 100) if len(raw_df) else 0
    n_cleaned = len(df)

    render_metric_row([
        ("Total Dataset", f"{len(raw_df):,}", "Ulasan terlabel", False),
        ("Data Unik Setelah Preprocessing", f"{n_cleaned:,}", "Deduplikasi konten + cleaning", False),
        ("Sentimen Positif", f"{n_pos:,}", f"{pct_pos:.1f}% dari total", False),
        ("Sentimen Negatif", f"{n_neg:,}", f"{100-pct_pos:.1f}% dari total", True),
    ])

    st.caption(
        f"Pipeline training menggunakan {n_cleaned:,} data unik setelah deduplikasi konten dan preprocessing."
    )

    if metadata:
        render_insight_card(
            title="Kesimpulan Training",
            content=metadata.get("conclusion", "Belum tersedia."),
            bar_pct=87,
            emoji="📊",
        )

    col_chart, col_table = st.columns(2, gap="medium", vertical_alignment="top")
    with col_chart:
        section_title("Distribusi Label Sentimen")
        with st.container(border=True, key="honda_home_chart"):
            show_plotly_chart(
                plot_sentiment_distribution(df, chart_height=360),
                key="home_dist_chart",
            )
    with col_table:
        section_title("Contoh Dataset — Sudah Dipreprocessing")
        with st.container(border=True, key="honda_home_table"):
            styled_dataframe(df[["content", "cleaned_text", "label"]].head(12), height=420)


def page_visualization(
    df: pd.DataFrame, cm_nb, cm_svm, class_names: List[str], metrics_df: pd.DataFrame
) -> None:
    row1_a, row1_b = st.columns(2)
    with row1_a:
        section_title("Distribusi Sentimen")
        show_plotly_chart(plot_sentiment_distribution(df, "Overview Sentimen"))
    with row1_b:
        if not metrics_df.empty:
            section_title("Perbandingan Metrik")
            show_plotly_chart(plot_metrics_comparison(metrics_df))

    section_title("WordCloud — Pola Kata")
    pos_text = " ".join(df[df["label"] == "positif"]["cleaned_text"])
    neg_text = " ".join(df[df["label"] == "negatif"]["cleaned_text"])
    wc1, wc2 = st.columns(2)
    with wc1:
        st.pyplot(generate_wordcloud(pos_text, "Sentimen Positif", "Greens"))
    with wc2:
        st.pyplot(generate_wordcloud(neg_text, "Sentimen Negatif", "Reds"))

    section_title("Confusion Matrix")
    cm1, cm2 = st.columns(2)
    with cm1:
        show_plotly_chart(plot_confusion_matrix(cm_nb, class_names, "Naive Bayes"))
    with cm2:
        show_plotly_chart(plot_confusion_matrix(cm_svm, class_names, "SVM (LinearSVC)"))

    if not metrics_df.empty:
        section_title("Detail Metrik per Model")
        for metric in ["Accuracy", "Precision", "Recall", "F1-score"]:
            sub = metrics_df[["Model", metric]]
            fig = px.bar(
                sub, x="Model", y=metric, text=metric,
                color="Model", color_discrete_sequence=[C_METAL, C_ACCENT],
            )
            fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig.update_layout(yaxis=dict(range=[0, 1.05]), showlegend=False)
            show_plotly_chart(apply_plotly_theme(fig, f"Perbandingan {metric}"))


def page_predict_text(nb_model, svm_model, tfidf, label_encoder) -> None:
    section_title("Prediksi Sentimen Kalimat Baru")
    st.markdown(
        '<div class="panel-white"><p style="color:#4F4F4F;margin:0;">'
        "Pilih model di bawah lalu masukkan teks ulasan. Teks akan diproses "
        "(cleaning, normalisasi, stemming) dan diklasifikasikan.</p></div>",
        unsafe_allow_html=True,
    )

    model_name = st.selectbox(
        "Model Prediksi",
        ["Naive Bayes", "SVM"],
        index=1,
        label_visibility="collapsed",
        key="prediction_model_text",
    )
    model = nb_model if model_name == "Naive Bayes" else svm_model

    user_input = st.text_area(
        "Teks ulasan", height=140, placeholder="Contoh: Aplikasi sangat membantu dan mudah digunakan…",
        label_visibility="collapsed",
    )

    if st.button("Jalankan Prediksi", type="primary", use_container_width=False):
        if not user_input.strip():
            st.warning("Silakan masukkan teks terlebih dahulu.")
        else:
            with st.spinner("Memproses teks & memprediksi…"):
                time.sleep(0.35)
                cleaned = preprocess_text(user_input)
                pred_label = predict_sentiment(user_input, cleaned, model, tfidf, label_encoder)

            render_prediction_result(cleaned, str(pred_label), model_name)

    section_title("Contoh Prediksi Otomatis")
    examples = [
        "Aplikasi sangat membantu dan mudah digunakan",
        "Aplikasi error dan sangat mengecewakan",
    ]
    ex_df = pd.DataFrame({"content": examples})
    ex_df["cleaned_text"] = ex_df["content"].apply(preprocess_text)
    ex_df["prediction"] = ex_df.apply(
        lambda row: predict_sentiment(str(row["content"]), str(row["cleaned_text"]), model, tfidf, label_encoder),
        axis=1,
    )
    styled_dataframe(ex_df)


def page_predict_csv(nb_model, svm_model, tfidf, label_encoder) -> None:
    section_title("Prediksi Sentimen Massal (CSV)")
    st.caption("File wajib memiliki kolom `content`.")

    model_name = st.selectbox(
        "Model Prediksi",
        ["Naive Bayes", "SVM"],
        index=1,
        label_visibility="collapsed",
        key="prediction_model_csv",
    )
    model = nb_model if model_name == "Naive Bayes" else svm_model

    uploaded_csv = st.file_uploader(
        "Upload CSV untuk prediksi massal",
        type=["csv"],
        label_visibility="collapsed",
        key="csv_prediction_upload",
    )

    if uploaded_csv is None:
        st.info("Upload file CSV di sini untuk memulai prediksi massal.")
        return

    try:
        pred_df = pd.read_csv(uploaded_csv)
        if "content" not in pred_df.columns:
            st.error("Kolom `content` tidak ditemukan pada file.")
            return

        progress = st.progress(0, text="Membaca file…")
        progress.progress(15)
        pred_df["cleaned_text"] = pred_df["content"].astype(str).apply(preprocess_text)
        progress.progress(55, text="Vektorisasi TF-IDF…")
        pred_vec = tfidf.transform(pred_df["cleaned_text"])
        progress.progress(80, text="Memprediksi sentimen…")
        pred_df["predicted_label"] = label_encoder.inverse_transform(model.predict(pred_vec))
        progress.progress(100, text="Selesai!")
        time.sleep(0.25)
        progress.empty()

        st.success(f"Prediksi massal selesai — {len(pred_df):,} baris diproses.")
        styled_dataframe(pred_df.head(25), height=420)

        st.download_button(
            label="⬇ Download Hasil Prediksi CSV",
            data=pred_df.to_csv(index=False).encode("utf-8"),
            file_name="hasil_prediksi_sentimen.csv",
            mime="text/csv",
            use_container_width=True,
        )
    except Exception as exc:
        st.error(f"Terjadi error: {exc}")


def page_evaluation(metrics_df: pd.DataFrame, cv_df: pd.DataFrame, metadata: Dict) -> None:
    section_title("Ringkasan Evaluasi Model")

    if metrics_df.empty:
        st.warning("File evaluasi belum ditemukan. Jalankan `python train_model.py` terlebih dahulu.")
        return

    for _, row in metrics_df.iterrows():
        st.markdown(f"#### {row['Model']}")
        render_metric_row([
            ("Accuracy", f"{row['Accuracy']:.2%}", "Akurasi test set", False),
            ("Precision", f"{row['Precision']:.2%}", "Presisi", False),
            ("Recall", f"{row['Recall']:.2%}", "Recall", False),
            ("F1-Score", f"{row['F1-score']:.2%}", "Harmonic mean", row["Model"].startswith("SVM")),
        ])

    col_t, col_c = st.columns([1.2, 1])
    with col_t:
        section_title("Tabel Evaluasi")
        display = metrics_df.copy()
        for c in ["Accuracy", "Precision", "Recall", "F1-score"]:
            display[c] = display[c].map(lambda x: f"{x:.4f}")
        styled_dataframe(display)

    with col_c:
        if not cv_df.empty:
            section_title("Cross-Validation (5-Fold)")
            show_plotly_chart(plot_cv_results(cv_df))
            styled_dataframe(cv_df, height=180)

    if metadata:
        section_title("Kesimpulan Otomatis")
        render_insight_card(
            title="Rekomendasi Model",
            content=metadata.get("conclusion", "Kesimpulan belum tersedia."),
            bar_pct=92,
            emoji="🏆",
        )


# =====================================================
# MAIN
# =====================================================
def load_all_data():
    """Memuat dataset & model dengan progress bar."""
    data_mtime = os.path.getmtime(DATA_PATH) if os.path.exists(DATA_PATH) else 0.0
    model_files = [
        "nb_model.pkl",
        "svm_model.pkl",
        "tfidf.pkl",
        "label_encoder.pkl",
        "preprocessor_config.json",
    ]
    artifact_signature = tuple(
        os.path.getmtime(os.path.join(MODELS_DIR, filename))
        if os.path.exists(os.path.join(MODELS_DIR, filename))
        else 0.0
        for filename in model_files
    )
    bar = st.progress(0, text="Inisialisasi dashboard…")
    bar.progress(10, text="Memuat dataset…")
    raw_df = load_dataset()
    df = load_dataset_with_cleaned_text(data_mtime)
    bar.progress(55, text="Memuat model ML…")
    models = load_models(artifact_signature)
    bar.progress(85, text="Menyiapkan evaluasi…")
    metrics_path = os.path.join(MODELS_DIR, "evaluation_metrics.csv")
    cv_path = os.path.join(MODELS_DIR, "cv_results.csv")
    cm_path = os.path.join(MODELS_DIR, "confusion_matrices.json")
    metrics_df = pd.read_csv(metrics_path) if os.path.exists(metrics_path) else pd.DataFrame()
    cv_df = pd.read_csv(cv_path) if os.path.exists(cv_path) else pd.DataFrame()
    cm_data: Dict = {}
    if os.path.exists(cm_path):
        with open(cm_path, "r", encoding="utf-8") as f:
            cm_data = json.load(f)
    metadata = {}
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    bar.progress(100, text="Siap!")
    time.sleep(0.2)
    bar.empty()
    return raw_df, df, models, metrics_df, cv_df, cm_data, metadata


def render_sidebar() -> str:
    with st.sidebar:
        # Make sidebar content breathe a bit more and feel premium
        st.markdown(
            f"""
            <div class="honda-sidebar-brand">
                <div class="honda-sidebar-title">HONDA</div>
                <div class="honda-sidebar-subtitle">E‑CARE ANALYTICS</div>
                <div class="honda-sidebar-divider"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="honda-sidebar-section">NAVIGATION</div>', unsafe_allow_html=True)

        nav_choice = st.radio(
            "Navigasi",
            NAV_LABELS,
            index=0,
            label_visibility="collapsed",
            key="honda_nav_menu",
        )
        menu = MENU_OPTIONS[NAV_LABELS.index(nav_choice)]

        st.markdown('<div class="honda-sidebar-spacer"></div>', unsafe_allow_html=True)
        st.markdown('<div class="honda-soft-divider"></div>', unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="honda-sidebar-meta">
                <div class="honda-sidebar-meta-row">Dataset: Honda e‑Care Reviews</div>
                <div class="honda-sidebar-meta-row">Label: <b>positif</b> · <b>negatif</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return menu



def main() -> None:

    inject_custom_css()
    menu = render_sidebar()
    render_header()

    try:
        with st.spinner("Menghubungkan ke pipeline analisis sentimen…"):
            raw_df, df, (nb_model, svm_model, tfidf, label_encoder), metrics_df, cv_df, cm_data, metadata = load_all_data()
    except Exception as exc:
        st.error(
            "Model belum tersedia. Jalankan perintah berikut dari folder `sentiment-analysis`:\n\n"
            "```bash\npython train_model.py\n```\n\n"
            f"Detail: {exc}"
        )
        st.stop()

    # Gunakan confusion matrix yang disimpan saat training (bukan recompute)
    # agar nilai di dashboard selalu cocok dengan reports/
    import numpy as np
    if cm_data:
        class_names = cm_data["class_names"]
        cm_nb = np.array(cm_data["cm_nb"])
        cm_svm = np.array(cm_data["cm_svm"])
    else:
        # Fallback: recompute jika file JSON belum ada (jalankan train_model.py dulu)
        class_names = list(label_encoder.classes_)
        y = label_encoder.transform(df["label"])
        X = tfidf.transform(df["cleaned_text"])
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        cm_nb = confusion_matrix(y_test, nb_model.predict(X_test))
        cm_svm = confusion_matrix(y_test, svm_model.predict(X_test))

    # Routing halaman
    if menu == "Halaman Utama":
        page_home(raw_df, df, metadata)
    elif menu == "Visualisasi":
        page_visualization(df, cm_nb, cm_svm, class_names, metrics_df)
    elif menu == "Prediksi Sentimen":
        page_predict_text(nb_model, svm_model, tfidf, label_encoder)
        page_predict_csv(nb_model, svm_model, tfidf, label_encoder)
    elif menu == "Hasil Evaluasi":
        page_evaluation(metrics_df, cv_df, metadata)


    render_footer()


if __name__ == "__main__":
    main()
