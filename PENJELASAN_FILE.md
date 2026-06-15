# Penjelasan Singkat File di Folder `sentiment-analysis`

Dokumen ini merangkum fungsi utama setiap file dan subfolder pada proyek analisis sentimen ulasan aplikasi **Honda e-Care**.

---

## File di akar folder

| File | Penjelasan singkat |
|------|-------------------|
| **`train_model.py`** | Skrip utama pipeline: membaca dataset, preprocessing teks Bahasa Indonesia, ekstraksi fitur TF-IDF, melatih model **Multinomial Naive Bayes** dan **SVM (LinearSVC)**, evaluasi di data uji, cross-validation, menyimpan model (`.pkl`), konfigurasi preprocessor, metrik, dan gambar laporan ke folder `models/` dan `reports/`. |
| **`dashboard.py`** | Aplikasi **Streamlit** interaktif: ringkasan data, visualisasi (termasuk Plotly), prediksi sentimen per teks, prediksi massal dari CSV, serta halaman hasil evaluasi model. |
| **`export_dashboard_corpus.py`** | Menghasilkan snapshot **`models/dashboard_corpus.csv`** (kolom `content`, `label`, `cleaned_text`) agar dashboard bisa memuat teks bersih tanpa menjalankan ulang preprocessing penuh pada ribuan baris—selama file CSV sumber belum berubah dari waktu training terakhir. |
| **`requirements.txt`** | Daftar dependensi Python yang diperlukan untuk menjalankan training dan dashboard. |
| **`README.md`** | Dokumentasi proyek: tujuan, struktur folder, dan cara menjalankan training serta dashboard. |
| **`PENJELASAN_FILE.md`** | File ini — ringkasan penjelasan struktur file proyek. |

---

## Folder `data/`

| File | Penjelasan singkat |
|------|-------------------|
| **`ulasan_honda_ecare_5k.csv`** | Dataset ulasan pengguna. Format yang didukung antara lain kolom **`content`** dan **`label`** (positif/negatif), atau **`content`** + **`score`** (skor dinormalisasi menjadi label). |

---

## Folder `models/` (hasil setelah menjalankan `train_model.py`)

| File | Penjelasan singkat |
|------|-------------------|
| **`nb_model.pkl`** | Model **Naive Bayes** terlatih (diserialisasi dengan `joblib`). |
| **`svm_model.pkl`** | Model **SVM (LinearSVC)** terlatih. |
| **`tfidf.pkl`** | Objek **TfidfVectorizer** yang sama dipakai saat training dan saat prediksi di dashboard. |
| **`label_encoder.pkl`** | Encoder label (misalnya pemetaan angka ↔ string `positif` / `negatif`). |
| **`preprocessor_config.json`** | Kamus normalisasi singkatan/kata informal dan daftar **stopword** agar preprocessing di dashboard konsisten dengan training. |
| **`evaluation_metrics.csv`** | Tabel metrik evaluasi per model pada **test set** (Accuracy, Precision, Recall, F1-score). |
| **`cv_results.csv`** | Ringkasan **cross-validation** (misalnya mean dan standar deviasi akurasi per model). |
| **`metadata.json`** | Ringkasan angka training (jumlah data sebelum/sesudah cleaning, ukuran train/test, model terbaik, kesimpulan teks, timestamp CSV sumber, dll.). |
| **`dashboard_corpus.csv`** | Snapshot hasil preprocessing untuk dashboard (opsional, diperbarui oleh `train_model.py` atau `export_dashboard_corpus.py`). |

---

## Folder `reports/` (hasil visual dari training)

Berisi **gambar PNG** yang disimpan oleh `train_model.py`, antara lain:

- Distribusi label, histogram panjang teks  
- Wordcloud sentimen positif/negatif  
- Confusion matrix per model (**`confusion_matrix_nb.png`**, **`confusion_matrix_svm.png`**, dll.)  
- Grafik perbandingan metrik antar model  

Nama file mengikuti yang ditulis di dalam skrip `train_model.py`.

---

## Alur kerja singkat

1. Siapkan / perbarui **`data/ulasan_honda_ecare_5k.csv`**.  
2. Jalankan **`python train_model.py`** → mengisi **`models/`** dan **`reports/`**.  
3. (Opsional) Jalankan **`python export_dashboard_corpus.py`** jika ingin mempercepat muat data di dashboard.  
4. Jalankan **`streamlit run dashboard.py`** untuk membuka dashboard analisis sentimen.

---

*Terakhir disesuaikan dengan struktur proyek sentiment-analysis.*
