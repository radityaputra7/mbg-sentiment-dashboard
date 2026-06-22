# Dashboard Evaluasi BGN — Analisis Sentimen Keracunan MBG di X

Revisi tugas besar **Kecerdasan Artifisial dan Penerapannya**.
Dashboard menerjemahkan output klasifikasi sentimen menjadi insight evaluasi konkret untuk Badan Gizi Nasional (BGN).

## Pipeline AI 4-Layer

1. **Layer 1 — Klasifikasi Sentimen** (Supervised): IndoBERT + TF-IDF/SVM → label final (negative / neutral / positive / uncertain).
2. **Layer 2 — Aspect Mapping** (Rule-based): 7 aspek operasional MBG, multi-label keyword matching → interpretable baseline.
3. **Layer 3 — Topic Discovery** (Unsupervised AI BARU): TF-IDF + Non-negative Matrix Factorization (NMF, 6 components) → menemukan tema diskursus tanpa pre-defined keywords.
4. **Layer 4 — Konvergensi**: cross-validation rule-based vs NMF → mengukur konsistensi dua pendekatan.

## Struktur dashboard

1. Ringkasan Distribusi Sentimen
2. Aspect Mapping Rule-based (7 aspek)
3. **Topic Discovery via NMF** — termasuk visualisasi 2D dan insight baru
4. **Konvergensi Aspek ↔ Topik** — heatmap cross-validation
5. Rekomendasi Evaluasi BGN (per aspek + tambahan dari NMF)
6. Eksplorasi Tweet (per aspek dan per topik)

## File dalam repo

| File | Ukuran | Deskripsi |
|---|---|---|
| `app.py` | 33 KB | Streamlit dashboard utama |
| `hasil_sentimen_slim.csv` | 20 MB | Data sentimen slim (60.616 tweet) |
| `topic_assignments.csv` | 6 MB | Dominant topic per tweet negatif (NMF) |
| `topic_metadata.csv` | 2 KB | Metadata 6 topik + interpretasi |
| `topic_2d_sample.csv` | 1 MB | 5.000 tweet diproyeksikan ke 2D untuk scatter |
| `requirements.txt` | <1 KB | Python dependencies |

Total: ~28 MB, aman untuk GitHub (limit 100 MB/file) dan Streamlit Cloud (RAM 1 GB).

## Cara menjalankan secara lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

Browser akan otomatis terbuka di `localhost:8501`.

## Cara deploy ke Streamlit Community Cloud

1. Push semua file ke repo GitHub (public atau private).
2. Buka https://share.streamlit.io
3. Login dengan GitHub.
4. **Create app** → pilih repo → main file: `app.py` → **Deploy**.
5. URL hasil: `https://<app-name>.streamlit.app`.

## Insight utama dari pipeline AI

**Yang ditemukan kedua pendekatan (konvergen):**
- Kualitas makanan basi & kondisi dapur (konvergen kuat antara aspek manual & topik NMF).

**Yang HANYA ditemukan NMF (insight baru):**
- **Plesetan 'Makan Beracun Gratis'** (9.4% tweet negatif) → reputational damage di level branding.
- **Persepsi 'berita keracunan tiap hari'** (6.8%) → news fatigue, butuh strategi komunikasi krisis.
- **Fokus diskursus pada angka korban** (6.6%) → publik mempertanyakan transparansi data resmi.

## Stack

- Python 3.10+
- Streamlit, pandas, plotly, numpy
- scikit-learn (untuk NMF, precomputed di Colab notebook)
