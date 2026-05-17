# MindTrack UB — Student Mental Health Analysis

**MindTrack UB** adalah prototype sistem pendukung keputusan berbasis data dan machine learning untuk membantu kampus membaca pola risiko kesehatan mental mahasiswa secara lebih awal. Project ini dikembangkan untuk kategori **AI & Data Innovation Challenge** dengan fokus pada analisis data, treatment gap, model screening, serta dashboard interaktif berbasis Streamlit.

Project ini **tidak bertujuan memberikan diagnosis medis**, tetapi digunakan sebagai proof of concept untuk menunjukkan bagaimana data dapat membantu kampus melakukan pemetaan risiko, membaca pola umum, dan menentukan tindak lanjut pendampingan secara lebih terarah.

---

## 1. Latar Belakang

Kesehatan mental mahasiswa menjadi isu penting dalam lingkungan kampus yang besar dan dinamis. Mahasiswa dapat mengalami tekanan akademik, tuntutan prestasi, adaptasi sosial, masalah ekonomi, hingga burnout.

Dalam konteks smart campus, data dapat dimanfaatkan untuk membantu kampus memahami pola risiko secara lebih adaptif. Oleh karena itu, project ini mencoba menjawab pertanyaan utama:

> Bagaimana data dan machine learning dapat membantu kampus melakukan screening awal terhadap risiko kesehatan mental mahasiswa tanpa menggantikan peran konselor?

---

## 2. Tujuan Project

Project ini memiliki beberapa tujuan utama:

1. Menganalisis pola indikator kesehatan mental mahasiswa.
2. Mengidentifikasi treatment gap, yaitu kondisi ketika mahasiswa memiliki indikator risiko tetapi belum mencari bantuan.
3. Membuat segmentasi risiko berdasarkan hasil analisis indikator mental health.
4. Melatih model machine learning untuk screening awal **No Risk vs At Risk**.
5. Membuat dashboard Streamlit sebagai prototype sistem pendukung keputusan.
6. Menjelaskan batasan, etika, dan risiko penggunaan AI dalam konteks kesehatan mental.

---

## 3. Dataset

Dataset yang digunakan:

```text
Student Mental health.csv
```

Dataset berisi data mahasiswa dengan atribut seperti:

- Gender
- Age
- Course
- Year of Study
- CGPA
- Marital Status
- Depression
- Anxiety
- Panic Attack
- Seek Specialist Treatment

Kolom `depression`, `anxiety`, dan `panic_attack` digunakan untuk membentuk indikator risiko, sedangkan model machine learning hanya memakai fitur non-klinis agar tidak terjadi data leakage.

---

## 4. Insight Utama

Dari hasil analisis dataset:

| Insight | Nilai |
|---|---:|
| Total responden | 101 |
| Mahasiswa dengan minimal 1 indikator mental health | 64 |
| Mahasiswa High Risk | 28 |
| Mahasiswa yang mencari bantuan spesialis | 6 |
| Treatment gap | 58 |

Insight paling penting:

> Banyak mahasiswa menunjukkan indikator risiko mental health, tetapi hanya sedikit yang mencari bantuan spesialis. Hal ini menunjukkan adanya treatment gap dan memperkuat kebutuhan sistem screening awal yang lebih proaktif.

---

## 5. Pendekatan Analisis

Project ini menggunakan dua pendekatan utama:

### 5.1 Analytical Risk Segmentation

Segmentasi ini digunakan untuk membaca pola data dan prioritas pendampingan berdasarkan jumlah indikator mental health.

```text
Low Risk    = 0 indikator
Medium Risk = 1 indikator
High Risk   = 2–3 indikator
```

Indikator yang dihitung:

- Depression
- Anxiety
- Panic Attack

Segmentasi ini digunakan pada bagian EDA, treatment gap analysis, dan visualisasi dashboard.

---

### 5.2 Machine Learning Screening

Model machine learning digunakan untuk melakukan screening awal dengan target binary:

```text
No Risk = tidak memiliki indikator mental health
At Risk = memiliki minimal 1 indikator mental health
```

Fitur yang digunakan untuk model:

- Gender
- Age
- Course
- Year of Study
- CGPA
- Marital Status

Fitur yang **tidak digunakan** sebagai input model:

- Depression
- Anxiety
- Panic Attack

Tiga fitur tersebut tidak digunakan karena merupakan pembentuk target risiko. Jika dimasukkan ke model, hasil evaluasi akan terlihat sangat tinggi tetapi tidak valid karena terjadi **data leakage**.

---

## 6. Hasil Modeling

Beberapa eksperimen model dilakukan, seperti:

- Logistic Regression
- Logistic Regression Balanced
- Decision Tree Balanced
- Random Forest Balanced
- Extra Trees Balanced
- Gradient Boosting
- Random Oversampling
- SMOTE
- Dummy Baseline
- Leakage Demo

Model valid terbaik:

| Model | Target | Accuracy CV | Balanced Accuracy CV | F1 Macro CV | F1 Weighted CV |
|---|---|---:|---:|---:|---:|
| Gradient Boosting | Binary At Risk | 0.702 | 0.633 | 0.630 | 0.674 |

Kesimpulan modeling:

> Model binary **No Risk vs At Risk** lebih stabil dibandingkan model multiclass Low / Medium / High Risk karena ukuran dataset kecil dan fitur yang tersedia masih terbatas.

Model ini diposisikan sebagai **proof of concept**, bukan sistem produksi.

---

## 7. Dashboard Streamlit

Dashboard dibuat menggunakan Streamlit dan terdiri dari beberapa bagian:

### 7.1 Executive Overview

Menampilkan ringkasan utama:

- Total responden
- Jumlah mahasiswa dengan indikator mental health
- Jumlah High Risk
- Jumlah mahasiswa yang mencari bantuan
- Treatment gap

### 7.2 Risk Segmentation

Menampilkan distribusi:

- Low Risk
- Medium Risk
- High Risk

Bagian ini berbasis hasil analisis indikator mental health, bukan prediksi model.

### 7.3 Treatment Gap Analysis

Menampilkan hubungan antara risk level dan status pencarian bantuan spesialis.

Fokus utama:

> Mahasiswa yang memiliki indikator risiko tetapi belum mencari bantuan.

### 7.4 Pattern Analysis

Menampilkan pola risiko berdasarkan:

- Gender
- Year of Study
- CGPA
- Course

Bagian ini hanya digunakan untuk membaca pola umum dan tidak boleh digunakan untuk menstigma kelompok tertentu.

### 7.5 Model Evidence

Menampilkan hasil eksperimen model, performa model terbaik, perbandingan eksperimen, serta penjelasan valid modeling vs leakage modeling.

### 7.6 ML Screening Simulation

Simulasi ini menggunakan model **Gradient Boosting** untuk menghasilkan skor estimasi terhadap kelas **At Risk**.

Input simulasi:

- Gender
- Age
- Course
- Year of Study
- CGPA
- Marital Status

Output dashboard:

```text
Estimated At-Risk Score
```

Skor kemudian dipetakan menjadi:

| Skor Estimasi At Risk | Kategori Dashboard |
|---:|---|
| < 50% | No Risk |
| 50% - 69.9% | Need Monitoring |
| >= 70% | At Risk |

Catatan:

> Need Monitoring bukan kelas asli model, tetapi zona interpretasi dari skor probabilitas agar hasil dashboard lebih aman dan tidak terlalu keras dalam pengambilan keputusan.

### 7.7 Ethics & Limits

Menjelaskan prinsip penggunaan sistem:

- Bukan diagnosis medis.
- Tidak menggantikan psikolog atau konselor.
- Tidak boleh digunakan untuk menghukum atau menstigma mahasiswa.
- Data sensitif harus dianonimkan.
- Keputusan akhir tetap berada pada manusia/konselor.

---

## 8. Struktur Project

```text
NMITGGSOG-STUDENT-MENTAL-HEALTH-ANALYSIS/
│
├── app/
│   └── app.py
│
├── data/
│   └── Student Mental health.csv
│
├── docs/
│   ├── solution_summary.md
│   └── worksheet_final.pdf
│
├── notebooks/
│   ├── 01_mindtrack_analysis_pipeline.ipynb
│   ├── 02_model_improvement_experiments.ipynb
│   └── 02_model_improvement_experiments_with_feature_importance.ipynb
│
├── outputs/
│   ├── dashboard_preview.png
│   ├── mental_health_indicators.png
│   ├── risk_distribution.png
│   ├── treatment_gap.png
│   ├── confusion_matrix.png
│   ├── model_metrics.csv
│   ├── leakage_demo_results.csv
│   ├── overall_valid_feature_importance.csv
│   └── overall_leakage_feature_importance.csv
│
├── README.md
└── requirements.txt
```

Folder `tekra/` adalah virtual environment lokal dan tidak perlu dimasukkan ke GitHub.

---

## 9. Cara Setup Project

### 9.1 Masuk ke folder project

```powershell
cd D:\Github\nmitggsog-student-mental-health-analysis
```

### 9.2 Buat virtual environment dengan uv

```powershell
uv venv tekra
```

### 9.3 Aktifkan virtual environment

```powershell
.\tekra\Scripts\activate
```

Jika PowerShell menolak aktivasi karena execution policy, jalankan:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Lalu aktifkan ulang:

```powershell
.\tekra\Scripts\activate
```

### 9.4 Install dependencies

```powershell
uv pip install -r requirements.txt
```

### 9.5 Jalankan dashboard

```powershell
streamlit run app/app.py
```

Atau jika command `streamlit` tidak terbaca:

```powershell
.\tekra\Scripts\python.exe -m streamlit run app/app.py
```

---

## 10. Requirements

Isi `requirements.txt`:

```text
streamlit
pandas
numpy
scikit-learn
imbalanced-learn
plotly
matplotlib
seaborn
jupyter
notebook
ipykernel
nbformat
kaleido
```

---

## 11. Cara Menjalankan Notebook

Agar environment `tekra` muncul sebagai kernel Jupyter:

```powershell
python -m ipykernel install --user --name tekra --display-name "Python (tekra)"
```

Lalu buka notebook di VS Code atau Jupyter dan pilih kernel:

```text
Python (tekra)
```

Notebook utama:

```text
notebooks/01_mindtrack_analysis_pipeline.ipynb
```

Notebook eksperimen model terbaru:

```text
notebooks/02_model_improvement_experiments_with_feature_importance.ipynb
```

---

## 12. Catatan Penting tentang Model

Model dalam project ini tidak boleh dianggap sebagai alat diagnosis kesehatan mental.

Model hanya digunakan sebagai:

- proof of concept,
- screening awal,
- pendukung dashboard,
- bahan argumentasi bahwa data dapat membantu proses pengambilan keputusan.

Keterbatasan model:

1. Dataset kecil.
2. Dataset bukan data mahasiswa UB secara langsung.
3. Fitur masih terbatas pada data demografis dan akademik sederhana.
4. Belum ada data presensi, LMS, keterlambatan tugas, partisipasi kampus, atau histori konseling.
5. Probabilitas model belum boleh dianggap sebagai probabilitas medis.
6. Implementasi nyata membutuhkan validasi etik, teknis, dan klinis.

---

## 13. Data Leakage Explanation

Data leakage terjadi ketika fitur yang digunakan model mengandung jawaban dari target yang ingin diprediksi.

Dalam project ini:

```text
risk_level / binary_risk dibentuk dari depression, anxiety, panic_attack
```

Karena itu, tiga fitur berikut tidak boleh digunakan sebagai input model:

```text
depression
anxiety
panic_attack
```

Jika tiga fitur tersebut dimasukkan, performa model akan terlihat sangat tinggi, tetapi hasil tersebut tidak valid.

---

## 14. Interpretasi Dashboard

Dashboard menggunakan dua lapisan analisis:

### Lapisan 1 — Analytical Segmentation

Digunakan untuk membaca pola data:

```text
Low Risk
Medium Risk
High Risk
```

Lapisan ini berdasarkan jumlah indikator mental health yang muncul pada dataset.

### Lapisan 2 — ML Screening Simulation

Digunakan untuk simulasi prediksi berbasis model:

```text
No Risk
Need Monitoring
At Risk
```

Model menghasilkan skor estimasi terhadap kelas **At Risk**, lalu dashboard memetakan skor tersebut menjadi kategori yang lebih mudah dipahami.

---

## 15. Etika Penggunaan

Sistem seperti MindTrack UB harus mengikuti prinsip berikut:

- Output sistem tidak boleh digunakan sebagai label permanen terhadap mahasiswa.
- Hasil model tidak boleh menjadi dasar hukuman akademik.
- Data mahasiswa harus dijaga kerahasiaannya.
- Mahasiswa perlu mengetahui tujuan penggunaan data.
- Sistem harus digunakan untuk membantu, bukan mengawasi secara berlebihan.
- Keputusan akhir harus tetap dilakukan oleh manusia, khususnya konselor atau pihak profesional.

---

## 16. Kesimpulan

MindTrack UB menunjukkan bahwa data dan machine learning dapat digunakan sebagai dasar awal untuk membangun sistem pendukung keputusan dalam konteks kesehatan mental mahasiswa.

Hasil analisis menunjukkan adanya treatment gap yang cukup jelas: banyak mahasiswa memiliki indikator risiko, tetapi hanya sedikit yang mencari bantuan spesialis.

Model Gradient Boosting digunakan sebagai proof of concept untuk screening awal **No Risk vs At Risk**, sedangkan dashboard membantu menyajikan insight, pola risiko, treatment gap, dan simulasi rekomendasi tindak lanjut.

Dengan pengembangan lebih lanjut menggunakan data kampus yang lebih lengkap, sistem seperti MindTrack UB berpotensi membantu kampus menjadi lebih responsif terhadap kesejahteraan mahasiswa.

---

## 17. Disclaimer

Project ini dibuat untuk kebutuhan lomba dan eksplorasi akademik.

MindTrack UB bukan alat diagnosis medis, bukan pengganti psikolog, dan bukan sistem produksi. Semua hasil analisis dan prediksi harus dipahami sebagai sinyal awal yang memerlukan validasi lanjutan oleh pihak profesional.