# MindTrack UB — Silent Struggle Student Mental Health Analysis

**MindTrack UB** adalah prototype **decision support system** berbasis data dan machine learning untuk membantu kampus membaca pola risiko kesehatan mental mahasiswa secara lebih awal.

Project ini berfokus pada konsep **Silent Struggle**, yaitu kondisi ketika mahasiswa sudah menunjukkan indikator risiko mental health, tetapi belum mencari bantuan atau dukungan spesialis.

Project ini dikembangkan untuk kebutuhan **AI & Data Innovation Challenge** dengan pendekatan:

- exploratory data analysis,
- silent struggle analysis,
- support gap analysis,
- priority segmentation,
- machine learning screening,
- feature importance,
- permutation importance,
- SHAP interpretability,
- dan dashboard interaktif berbasis Streamlit.

> MindTrack UB bukan alat diagnosis medis dan bukan pengganti psikolog/konselor. Output sistem hanya digunakan sebagai sinyal awal untuk membantu proses monitoring dan prioritisasi pendampingan.

---

## 1. Latar Belakang

Kesehatan mental mahasiswa menjadi isu penting dalam lingkungan kampus yang besar dan dinamis. Mahasiswa dapat mengalami tekanan akademik, tuntutan prestasi, adaptasi sosial, masalah ekonomi, burnout, hingga hambatan untuk mencari bantuan.

Dalam banyak kasus, mahasiswa yang memiliki indikasi masalah mental health tidak selalu langsung mencari bantuan. Sebagian dari mereka memilih diam, menunda, atau tidak mengetahui layanan yang dapat diakses.

Kondisi inilah yang dalam project ini disebut sebagai:

> **Silent Struggle** — mahasiswa memiliki indikator risiko mental health, tetapi belum mencari bantuan spesialis.

Dengan jumlah mahasiswa yang besar, kampus membutuhkan pendekatan berbasis data agar dapat membaca pola risiko secara lebih awal dan melakukan pendampingan secara lebih terarah.

---

## 2. Tujuan Project

Project ini bertujuan untuk:

1. Menganalisis pola indikator kesehatan mental mahasiswa.
2. Mengidentifikasi kelompok mahasiswa yang masuk kategori **At Risk**.
3. Mengukur **Support Gap**, yaitu proporsi mahasiswa berisiko yang belum mencari bantuan.
4. Mengidentifikasi fenomena **Silent Struggle**.
5. Membuat segmentasi prioritas pendampingan mahasiswa.
6. Melatih model machine learning untuk screening awal **No Risk vs At Risk**.
7. Menambahkan interpretability melalui feature importance, permutation importance, dan SHAP.
8. Membuat dashboard Streamlit sebagai prototype sistem pendukung keputusan.
9. Menjelaskan batasan etika, privasi, dan limitasi penggunaan AI pada isu kesehatan mental.

---

## 3. Dataset

Dataset utama berada pada folder:

```text
data/Student Mental health.csv
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

Kolom indikator mental health:

```text
depression
anxiety
panic_attack
```

digunakan untuk membentuk label risiko. Namun, ketiga kolom tersebut **tidak digunakan sebagai input model valid** karena dapat menyebabkan data leakage.

---

## 4. Definisi Konsep

### 4.1 Mental Health Indicator

Indikator mental health dihitung dari tiga variabel:

```text
Depression
Anxiety
Panic Attack
```

Jumlah indikator dihitung sebagai:

```text
symptom_count = depression_flag + anxiety_flag + panic_flag
```

---

### 4.2 Risk Level

Risk level digunakan untuk membaca tingkat indikator mental health pada data.

```text
Low Risk    = 0 indikator
Medium Risk = 1 indikator
High Risk   = 2–3 indikator
```

Risk level ini digunakan untuk analisis dan visualisasi, bukan untuk diagnosis klinis.

---

### 4.3 At Risk

Target binary untuk machine learning dibuat sebagai berikut:

```text
No Risk = tidak memiliki indikator mental health
At Risk = memiliki minimal 1 indikator mental health
```

Model machine learning memprediksi apakah profil mahasiswa masuk ke kelompok **No Risk** atau **At Risk** berdasarkan fitur non-klinis.

---

### 4.4 Silent Struggle

Silent Struggle adalah kondisi ketika mahasiswa:

```text
memiliki minimal 1 indikator mental health
dan
belum mencari bantuan spesialis
```

Secara logika:

```text
Silent Struggle = At Risk AND Seek Treatment = No
```

---

### 4.5 Support Gap

Support Gap adalah persentase mahasiswa At Risk yang belum mencari bantuan.

```text
Support Gap = Silent Struggle / At Risk
```

Dalam hasil analisis:

```text
Support Gap = 58 / 64 = 90.6%
```

Artinya, sebagian besar mahasiswa yang memiliki indikator risiko belum mencari bantuan spesialis.

---

### 4.6 Priority Segment

Priority segment digunakan untuk membantu kampus membaca prioritas pendampingan.

```text
Low Priority     = 0 indikator
Need Monitoring  = 1 indikator dan belum mencari bantuan
High Priority    = 2–3 indikator dan belum mencari bantuan
Reached Support  = berisiko dan sudah mencari bantuan
```

Segmentasi ini membantu dashboard menampilkan kelompok mana yang perlu dimonitor lebih lanjut.

---

## 5. Insight Utama

Berdasarkan hasil analisis dataset:

| Insight | Nilai |
|---|---:|
| Total responden | 101 |
| At Risk / memiliki minimal 1 indikator | 64 |
| Silent Struggle | 58 |
| Reached Support | 6 |
| Support Gap | 90.6% |
| High Priority | 22 |

Insight utama:

> Masalah utama bukan hanya banyaknya mahasiswa yang memiliki indikator risiko, tetapi adanya kelompok **Silent Struggle**, yaitu mahasiswa yang sudah menunjukkan indikator mental health namun belum mencari bantuan.

---

## 6. Pendekatan Analisis

Project ini menggunakan beberapa lapisan analisis.

### 6.1 Exploratory Data Analysis

EDA dilakukan untuk melihat:

- distribusi indikator mental health,
- distribusi risk level,
- proporsi At Risk,
- proporsi Silent Struggle,
- support gap,
- pola berdasarkan gender,
- pola berdasarkan year of study,
- pola berdasarkan CGPA,
- pola berdasarkan course.

---

### 6.2 Support Gap Analysis

Support gap analysis digunakan untuk menjawab:

> Dari mahasiswa yang sudah memiliki indikator risiko, berapa banyak yang belum mencari bantuan?

Analisis ini penting karena menunjukkan bahwa layanan bantuan yang tersedia belum tentu otomatis diakses oleh mahasiswa yang membutuhkan.

---

### 6.3 Priority Segmentation

Priority segmentation membantu kampus membaca kelompok mahasiswa berdasarkan tingkat urgensi.

| Segment | Makna |
|---|---|
| Low Priority | Tidak ada indikator mental health |
| Need Monitoring | Ada 1 indikator dan belum mencari bantuan |
| High Priority | Ada 2–3 indikator dan belum mencari bantuan |
| Reached Support | Sudah memiliki indikator dan sudah mencari bantuan |

---

## 7. Pendekatan Machine Learning

Model machine learning digunakan untuk screening awal dengan target binary:

```text
No Risk vs At Risk
```

### 7.1 Fitur Model Valid

Fitur yang digunakan:

```text
gender
age
course
year_of_study
cgpa
marital_status
```

Fitur ini dipilih karena merupakan fitur non-klinis yang tersedia pada dataset.

---

### 7.2 Fitur yang Tidak Digunakan

Fitur berikut tidak digunakan sebagai input model valid:

```text
depression
anxiety
panic_attack
```

Alasannya:

```text
risk label dibentuk dari depression, anxiety, dan panic_attack
```

Jika ketiga fitur tersebut dimasukkan ke model, maka model akan mendapatkan jawaban secara langsung. Hal ini disebut **data leakage**.

---

### 7.3 Model yang Dicoba

Beberapa eksperimen model dilakukan:

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

---

## 8. Hasil Modeling

Model valid terbaik berdasarkan **F1 Macro**:

| Model | Experiment | Accuracy CV | Balanced Accuracy CV | F1 Macro CV | F1 Weighted CV |
|---|---|---:|---:|---:|---:|
| Logistic Regression Balanced + RandomOverSampler | Binary At Risk with Oversampling | 0.662 | 0.635 | 0.633 | 0.650 |

Model ini dipilih karena **F1 Macro** lebih cocok untuk membaca performa antar kelas pada dataset kecil dan tidak seimbang.

Selain binary model, notebook juga menyertakan perbandingan tambahan untuk beberapa target lain seperti **support status** dan **priority segment** sebagai eksplorasi model pendukung.

---

## 9. Interpretability

Project ini menyertakan beberapa pendekatan interpretability agar hasil model tidak hanya berupa prediksi, tetapi juga dapat dijelaskan secara lebih transparan.

### 9.1 Feature Importance

Feature importance digunakan untuk melihat fitur yang paling banyak membantu model dalam proses prediksi.

Output terkait:

```text
prediction_pipeline_outputs/tables/valid_feature_importance_summary.csv
prediction_pipeline_outputs/tables/overall_valid_feature_importance.csv
prediction_pipeline_outputs/tables/feature_importance_support_status.csv
```

---

### 9.2 Permutation Importance

Permutation importance digunakan untuk melihat penurunan performa model ketika satu fitur diacak.

Output terkait:

```text
prediction_pipeline_outputs/tables/best_binary_permutation_importance.csv
```

---

### 9.3 SHAP Analysis

SHAP digunakan untuk menjelaskan kontribusi fitur terhadap output model.

Output SHAP disimpan pada:

```text
prediction_pipeline_outputs/tables/shap_encoded_importance_best_binary.csv
prediction_pipeline_outputs/tables/shap_base_importance_best_binary.csv
```

Visualisasi SHAP disimpan pada:

```text
prediction_pipeline_outputs/visualizations/11_shap_base_feature_importance.png
prediction_pipeline_outputs/visualizations/12_shap_encoded_feature_importance_top15.png
prediction_pipeline_outputs/visualizations/13_shap_summary_plot_best_binary.png
```

Jika library SHAP belum tersedia, jalankan:

```powershell
uv pip install shap
```

atau pastikan `shap` sudah ada pada `requirements.txt`.

---

## 10. Dashboard Streamlit

Dashboard berada pada:

```text
app/app.py
```

Dashboard terdiri dari beberapa tab utama.

---

### 10.1 Executive Overview

Menampilkan ringkasan utama:

- Total responden
- At Risk
- Silent Struggle
- Reached Support
- Support Gap

Bagian ini juga menampilkan prevalensi indikator mental health dan komposisi support gap pada kelompok At Risk.

---

### 10.2 Risk & Priority

Menampilkan:

- jumlah mahasiswa per priority segment,
- support rate per risk level,
- tabel ringkasan risk level.

Segmentasi yang digunakan:

```text
Low Priority
Need Monitoring
High Priority
Reached Support
```

---

### 10.3 Support Gap

Menampilkan hubungan antara risk level dan support status.

Support status terdiri dari:

```text
No Indicator
Silent Struggle
Reached Support
```

Bagian ini membantu menunjukkan bahwa sebagian mahasiswa yang berisiko belum mencari bantuan.

---

### 10.4 Pattern Analysis

Menampilkan pola priority segment berdasarkan:

- gender,
- year of study,
- CGPA,
- course.

Catatan penting:

> Pola pada grafik tidak boleh digunakan untuk menstigma gender, jurusan, atau angkatan tertentu. Analisis ini hanya membantu membaca area yang mungkin membutuhkan dukungan lebih lanjut.

---

### 10.5 Model Evidence

Menampilkan:

- model terbaik,
- metrik evaluasi,
- perbandingan model,
- alasan binary model lebih dipilih,
- penjelasan valid modeling vs leakage modeling.

---

### 10.6 ML Screening Simulation

Simulasi ini menggunakan model:

```text
Logistic Regression Balanced + RandomOverSampler
```

Input simulasi:

- Gender
- Age
- Course
- Year of Study
- CGPA
- Marital Status

Output:

```text
Estimated At-Risk Score
```

Skor tersebut dipetakan menjadi:

| Estimated At-Risk Score | Kategori Dashboard |
|---:|---|
| < 50% | No Risk |
| 50% - 69.9% | Need Monitoring |
| >= 70% | At Risk |

Catatan:

> Need Monitoring bukan kelas asli model, tetapi zona interpretasi dari skor probabilitas agar hasil dashboard lebih aman digunakan sebagai pendukung keputusan.

---

### 10.7 Ethics & Limits

Menjelaskan batasan dan etika penggunaan sistem:

- bukan diagnosis medis,
- bukan pengganti konselor,
- tidak boleh digunakan untuk menghukum mahasiswa,
- data sensitif harus dijaga,
- keputusan akhir tetap berada pada manusia/konselor.

---

## 11. Struktur Project

```text
NMITGGSOG-STUDENT-MENTAL-HEALTH-ANALYSIS/
│
├── app/
│   └── app.py
│
├── data/
│   └── Student Mental health.csv
│
├── notebooks/
│   └── MindTrackUB_Silent_Struggle_AI_Data_Pipeline.ipynb
│
├── prediction_pipeline_outputs/
│   ├── predictions/
│   │   ├── student_mental_health_labeled.csv
│   │   └── student_mental_health_predictions.csv
│   │
│   ├── tables/
│   │   ├── age_summary.csv
│   │   ├── association_summary.csv
│   │   ├── best_binary_permutation_importance.csv
│   │   ├── cgpa_summary.csv
│   │   ├── combination_summary.csv
│   │   ├── course_issue_summary.csv
│   │   ├── feature_importance_support_status.csv
│   │   ├── gender_summary.csv
│   │   ├── indicator_summary.csv
│   │   ├── issue_count_summary.csv
│   │   ├── leakage_demo_results.csv
│   │   ├── leakage_feature_importance_summary.csv
│   │   ├── model_comparison_all_targets.csv
│   │   ├── model_comparison_priority_segment.csv
│   │   ├── model_comparison_support_status.csv
│   │   ├── model_improvement_experiments.csv
│   │   ├── overall_leakage_feature_importance.csv
│   │   ├── overall_valid_feature_importance.csv
│   │   ├── priority_segment_distribution.csv
│   │   ├── priority_summary.csv
│   │   ├── shap_base_importance_best_binary.csv
│   │   ├── shap_encoded_importance_best_binary.csv
│   │   ├── student_mental_health_cleaned.csv
│   │   ├── support_by_issue_count.csv
│   │   ├── support_status_distribution.csv
│   │   ├── support_status_summary.csv
│   │   ├── valid_feature_importance_summary.csv
│   │   └── year_summary.csv
│   │
│   └── visualizations/
│       ├── 01_kpi_cards_support_gap.png
│       ├── 01_label_distribution.png
│       ├── 02_lollipop_prevalensi_indikator.png
│       ├── 02_model_comparison_weighted_f1.png
│       ├── 03_donut_support_gap.png
│       ├── 04_funnel_silent_struggle.png
│       ├── 05_priority_segment_mapping.png
│       ├── 06_heatmap_cooccurrence_indikator.png
│       ├── 07_heatmap_tahun_cgpa_any_issue.png
│       ├── 08_average_valid_feature_importance.png
│       ├── 09_best_binary_permutation_importance.png
│       ├── 10_average_leakage_feature_importance.png
│       ├── 11_shap_base_feature_importance.png
│       ├── 12_shap_encoded_feature_importance_top15.png
│       ├── 13_shap_summary_plot_best_binary.png
│       ├── confusion_matrix_priority_segment.png
│       ├── confusion_matrix_support_status.png
│       └── feature_importance_support_status.png
│
├── .gitignore
├── README.md
└── requirements.txt
```

> Catatan: folder `tekra/` adalah virtual environment lokal dan sudah diabaikan melalui `.gitignore`, sehingga tidak perlu ikut di-push ke repository.

---

## 12. Cara Setup Project

### 12.1 Clone repository

```powershell
git clone <repository-url>
cd nmitggsog-student-mental-health-analysis
```

Jika repository sudah ada di lokal:

```powershell
cd D:\Github\nmitggsog-student-mental-health-analysis
```

---

### 12.2 Buat virtual environment dengan uv

```powershell
uv venv tekra
```

---

### 12.3 Aktifkan virtual environment

```powershell
.\tekra\Scripts\activate
```

Jika PowerShell menolak aktivasi karena execution policy:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Lalu aktifkan ulang:

```powershell
.\tekra\Scripts\activate
```

---

### 12.4 Install dependencies

```powershell
uv pip install -r requirements.txt
```

---

### 12.5 Jalankan dashboard

```powershell
streamlit run app/app.py
```

Jika command `streamlit` tidak terbaca:

```powershell
.\tekra\Scripts\python.exe -m streamlit run app/app.py
```

---

## 13. Requirements

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
shap
```

---

## 14. Cara Menjalankan Notebook

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
notebooks/MindTrackUB_Silent_Struggle_AI_Data_Pipeline.ipynb
```

Notebook ini menghasilkan output ke folder:

```text
prediction_pipeline_outputs/
```

---

## 15. Output Notebook

Notebook menghasilkan tiga jenis output utama.

### 15.1 Tables

Berada pada:

```text
prediction_pipeline_outputs/tables/
```

Output tabel berisi ringkasan EDA, support gap, priority segment, hasil eksperimen model, leakage demo, feature importance, permutation importance, dan SHAP.

Contoh file penting:

```text
student_mental_health_cleaned.csv
indicator_summary.csv
support_status_summary.csv
priority_summary.csv
model_improvement_experiments.csv
model_comparison_all_targets.csv
model_comparison_priority_segment.csv
model_comparison_support_status.csv
valid_feature_importance_summary.csv
overall_valid_feature_importance.csv
best_binary_permutation_importance.csv
shap_base_importance_best_binary.csv
shap_encoded_importance_best_binary.csv
```

---

### 15.2 Visualizations

Berada pada:

```text
prediction_pipeline_outputs/visualizations/
```

Visualisasi mencakup KPI support gap, prevalensi indikator, donut support gap, funnel silent struggle, priority segment mapping, heatmap, feature importance, SHAP plot, dan confusion matrix.

Contoh file penting:

```text
01_kpi_cards_support_gap.png
02_lollipop_prevalensi_indikator.png
03_donut_support_gap.png
04_funnel_silent_struggle.png
05_priority_segment_mapping.png
08_average_valid_feature_importance.png
09_best_binary_permutation_importance.png
11_shap_base_feature_importance.png
12_shap_encoded_feature_importance_top15.png
13_shap_summary_plot_best_binary.png
confusion_matrix_priority_segment.png
confusion_matrix_support_status.png
```

---

### 15.3 Predictions

Berada pada:

```text
prediction_pipeline_outputs/predictions/
```

File prediksi/labeled data:

```text
student_mental_health_labeled.csv
student_mental_health_predictions.csv
```

---

## 16. Catatan tentang Data Leakage

Data leakage terjadi ketika fitur yang digunakan model mengandung jawaban dari target yang ingin diprediksi.

Dalam project ini:

```text
binary_risk / risk_level dibentuk dari depression, anxiety, panic_attack
```

Karena itu, tiga fitur berikut tidak boleh digunakan sebagai input model valid:

```text
depression
anxiety
panic_attack
```

Jika tiga fitur tersebut dimasukkan ke model, performa akan terlihat sangat tinggi, tetapi hasil tersebut tidak valid.

Bagian leakage demo pada notebook hanya digunakan sebagai pembanding untuk menunjukkan mengapa penggunaan fitur pembentuk target tidak boleh dianggap sebagai evaluasi model yang sah.

---

## 17. Interpretasi Model

Model dalam project ini tidak dimaksudkan sebagai alat diagnosis.

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

## 18. Etika Penggunaan

Sistem seperti MindTrack UB harus mengikuti prinsip berikut:

- Output sistem tidak boleh digunakan sebagai label permanen terhadap mahasiswa.
- Hasil model tidak boleh menjadi dasar hukuman akademik.
- Data mahasiswa harus dijaga kerahasiaannya.
- Mahasiswa perlu mengetahui tujuan penggunaan data.
- Sistem harus digunakan untuk membantu, bukan mengawasi secara berlebihan.
- Keputusan akhir harus tetap dilakukan oleh manusia, khususnya konselor atau pihak profesional.

---

## 19. Rekomendasi Pengembangan Lanjutan

Agar sistem lebih kuat, pengembangan selanjutnya dapat menambahkan data kampus yang lebih relevan, seperti:

- data presensi,
- aktivitas LMS,
- keterlambatan pengumpulan tugas,
- performa akademik historis,
- partisipasi kegiatan kampus,
- histori penggunaan layanan konseling,
- survei wellbeing berkala,
- data self-assessment yang disetujui mahasiswa.

Dengan data yang lebih lengkap dan validasi etik yang baik, sistem seperti MindTrack UB dapat dikembangkan menjadi decision support system yang lebih akurat dan aman.

---

## 20. Kesimpulan

MindTrack UB menunjukkan bahwa data dan machine learning dapat digunakan sebagai dasar awal untuk membangun sistem pendukung keputusan dalam konteks kesehatan mental mahasiswa.

Hasil analisis menunjukkan adanya **Support Gap** yang besar: banyak mahasiswa memiliki indikator risiko, tetapi belum mencari bantuan spesialis. Fenomena ini dirangkum sebagai **Silent Struggle**.

Model **Logistic Regression Balanced + RandomOverSampler** digunakan sebagai proof of concept untuk screening awal **No Risk vs At Risk**, sedangkan dashboard membantu menyajikan insight, pola risiko, support gap, priority segment, dan simulasi rekomendasi tindak lanjut.

---

## 21. Disclaimer

Project ini dibuat untuk kebutuhan lomba dan eksplorasi akademik.

MindTrack UB bukan alat diagnosis medis, bukan pengganti psikolog, dan bukan sistem produksi. Semua hasil analisis dan prediksi harus dipahami sebagai sinyal awal yang memerlukan validasi lanjutan oleh pihak profesional.