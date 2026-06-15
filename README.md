# Sentiment Analysis Honda e-Care

Project ini merupakan implementasi analisis sentimen ulasan aplikasi **Honda e-Care** untuk kebutuhan penelitian skripsi. Sistem membandingkan performa dua model klasifikasi:

- **Multinomial Naive Bayes**
- **SVM (LinearSVC)**

Pipeline mencakup data cleaning, preprocessing teks Bahasa Indonesia, ekstraksi fitur TF-IDF, evaluasi model, cross-validation, penyimpanan model, dan dashboard interaktif Streamlit.

## Struktur Folder Project

```text
sentiment-analysis/
│
├── data/
│   └── ulasan_honda_ecare_5k.csv
│
├── models/
│   ├── nb_model.pkl
│   ├── svm_model.pkl
│   ├── tfidf.pkl
│   ├── label_encoder.pkl
│   ├── preprocessor_config.json
│   ├── evaluation_metrics.csv
│   ├── cv_results.csv
│   └── metadata.json
│
├── reports/
│   ├── countplot_label.png
│   ├── histogram_panjang_teks.png
│   ├── wordcloud_positif.png
│   ├── wordcloud_negatif.png
│   ├── confusion_matrix_nb.png
│   ├── confusion_matrix_svm.png
│   └── perbandingan_*.png
│
├── train_model.py
├── dashboard.py
├── requirements.txt
└── README.md
```

## Cara Install

1. Masuk ke folder project:
   ```bash
   cd sentiment-analysis
   ```
2. (Opsional) Buat virtual environment:
   ```bash
   python -m venv .venv
   ```
3. Aktifkan virtual environment:
   - Windows:
     ```bash
     .venv\Scripts\activate
     ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Cara Menjalankan Training Model

Jalankan perintah berikut:

```bash
python train_model.py
```

Script ini akan menjalankan proses berikut:

1. Load dataset dan cleaning (missing value, duplikat, reset index)
2. EDA (distribusi label, persentase, panjang teks)
3. Preprocessing teks lengkap:
   - case folding
   - cleaning (URL, HTML, mention, hashtag, angka, punctuation, emoji, non alfabet, karakter berulang)
   - normalisasi kata informal
   - tokenizing
   - stopword removal (NLTK Indonesia)
   - stemming (Sastrawi)
4. WordCloud sentimen positif dan negatif
5. Label encoding (`positif=1`, `negatif=0`)
6. TF-IDF (`max_features=5000`, `ngram_range=(1,2)`)
7. Split data 80:20 (stratify, random_state=42)
8. Training model Naive Bayes dan SVM
9. Evaluasi lengkap (confusion matrix, accuracy, precision, recall, F1-score, classification report, train/test accuracy)
10. Visualisasi evaluasi dan perbandingan metrik
11. Cross validation 5-fold
12. Kesimpulan model terbaik otomatis
13. Prediksi contoh kalimat baru
14. Simpan model dan artefak evaluasi

## Cara Menjalankan Dashboard Streamlit

Pastikan training sudah dijalankan minimal satu kali, lalu jalankan:

```bash
streamlit run dashboard.py
```

## Fitur Dashboard

### Sidebar
- Menu navigasi halaman
- Upload file CSV
- Pilihan model prediksi (Naive Bayes / SVM)

### Halaman Utama
- Judul aplikasi dan deskripsi penelitian
- Jumlah dataset
- Distribusi label
- Contoh data ulasan

### Visualisasi
- Grafik distribusi sentimen
- WordCloud positif
- WordCloud negatif
- Confusion Matrix Naive Bayes dan SVM
- Grafik perbandingan metrik model

### Prediksi Sentimen
- Input teks manual
- Tombol prediksi
- Menampilkan hasil preprocessing dan label prediksi

### Prediksi File CSV
- Upload file CSV baru (wajib kolom `content`)
- Prediksi massal
- Preview hasil prediksi
- Download hasil prediksi dalam CSV

### Hasil Evaluasi
- Metric card (accuracy, precision, recall, F1-score)
- Tabel evaluasi model
- Hasil cross-validation
- Kesimpulan otomatis model terbaik

## Catatan Akademik

- Dataset target: ulasan aplikasi Honda e-Care
- Label sentimen: `positif`, `negatif`
- Pendekatan ini cocok sebagai baseline kuat untuk penelitian skripsi text mining/analisis sentimen.
- Anda dapat menambahkan eksperimen lanjutan (mis. tuning hyperparameter, balancing data, dan uji model lain) sebagai pengembangan penelitian.
