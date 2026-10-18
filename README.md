# UTS Machine Learning – Linear Regression & Logistic Regression

- **Nama** : Martinianus Aristo Babor
- **NIM** : 241011402129
- **Kelas** : Reg B

## Isi Repositori

| File | Keterangan |
| --- | --- |
| `tugas_UTS.ipynb` | Notebook jawaban, sudah dijalankan (Run All) lengkap dengan output dan grafik |
| `Ads_Data.csv` | Dataset biaya iklan vs penjualan (40 data) |
| `Email_Data.csv` | Dataset email spam (60 data) |

Model dan metrik evaluasi diimplementasikan sendiri dengan NumPy (`numpy`, `pandas`, `matplotlib`). scikit-learn hanya dipakai pada soal Bonus.

## Ringkasan Hasil (`RANDOM_STATE = 29`)

### Bagian A – Linear Regression (Ads_Data, split 80/20)

- Persamaan (normal equation): **ŷ = 27,9234 + 9,0921 · BiayaIklan**. Setiap kenaikan biaya iklan Rp1 juta, penjualan naik sekitar Rp9,09 juta.

| Data | MAE | MSE | RMSE | R² |
| --- | --- | --- | --- | --- |
| Training | 4,4583 | 37,4644 | 6,1208 | 0,9378 |
| Testing | 4,9381 | 31,0552 | 5,5727 | 0,9590 |

- Gradient descent (5000 iterasi): lr = 0,001 terlalu lambat (b = 24,66; m = 9,61); lr = 0,02 pas dan sama dengan normal equation; lr = 0,1 divergen (overflow, hasil NaN).
- Prediksi: 4,5 juta → 68,84 juta; 7 juta → 91,57 juta; 12 juta → 137,03 juta (ekstrapolasi, kurang bisa dipercaya).

### Bagian B – Logistic Regression (Email_Data, split 75/25, lr = 0,1, 10000 iterasi)

- Persamaan: **P(Spam) = σ(−11,0835 + 0,9317 · JumlahLink + 0,9637 · KataPromo)**.
- Evaluasi data testing (15 email): accuracy 0,80; precision 0,857; recall 0,75; F1 0,80; confusion matrix TN = 6, FP = 1, FN = 2, TP = 6.
- Decision boundary: x₂ = 11,50 − 0,9668 · x₁. Ada 3 email testing yang salah diklasifikasikan.

| Threshold | FP | FN | Precision | Recall |
| --- | --- | --- | --- | --- |
| 0,3 | 3 | 2 | 0,667 | 0,75 |
| 0,5 | 1 | 2 | 0,857 | 0,75 |
| 0,7 | 1 | 2 | 0,857 | 0,75 |

- Threshold yang dipilih: **0,7** (FP lebih merugikan pengguna).
- Prediksi email baru (threshold 0,7): E1 Bukan Spam, E2 Bukan Spam, E3 Spam, E4 Spam.

### Bagian C & Bonus

- Jawaban konsep C1–C5 ada di notebook.
- Bonus: scikit-learn memberi hasil identik untuk Linear Regression. Untuk Logistic Regression koefisien berbeda karena scikit-learn memakai regularisasi L2 (`C = 1.0`) secara default dan solver yang berbeda; akurasi testing sama (0,80).
