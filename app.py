"""
Dashboard Evaluasi BGN
Analisis Sentimen Keracunan MBG di Platform X
=============================================
Tugas Besar Kecerdasan Artifisial dan Penerapannya

Pipeline AI:
1. IndoBERT + TF-IDF/SVM  → klasifikasi sentimen (sudah ditrain di notebook)
2. Rule-based Aspect Mapping → 7 aspek operasional (interpretable baseline)
3. NMF Topic Modeling      → unsupervised topic discovery (data-driven insight)
4. Konvergensi Aspek ↔ Topik → validasi dua pendekatan
"""

import ast
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# =====================================================================
# KONFIGURASI HALAMAN
# =====================================================================
st.set_page_config(
    page_title="Dashboard Evaluasi BGN — Sentimen MBG",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSV_PATH = "hasil_sentimen_slim.csv"
TOPIC_ASSIGN_PATH = "topic_assignments.csv"
TOPIC_META_PATH = "topic_metadata.csv"
TOPIC_2D_PATH = "topic_2d_sample.csv"


# =====================================================================
# TAKSONOMI ASPEK (Rule-based, interpretable baseline)
# =====================================================================
ASPECTS = {
    "Kualitas Makanan & Bahan Baku": {
        "keywords": [
            "basi", "busuk", "mentah", "hambar", "bau", "expired", "kedaluwarsa",
            "kualitas", "mutu", "ayam", "ikan", "daging", "sayur", "telur",
            "matang", "asin", "amis", "rasa", "jelek", "murah",
        ],
        "finding": (
            "Banyak laporan menyebut makanan basi, mentah, atau berbau. "
            "Kata kunci 'basi' dan 'busuk' muncul ribuan kali dalam tweet negatif, "
            "kerap dikaitkan dengan bahan baku murah dan kualitas rendah."
        ),
        "recommendations": [
            "Tetapkan standar mutu bahan baku (HACCP / SNI) yang wajib dipenuhi vendor sebelum kontrak.",
            "Wajibkan uji organoleptik dan pengukuran suhu makanan saat tiba di sekolah.",
            "Audit acak terhadap pemasok bahan baku minimal 1x per bulan oleh tim independen.",
            "Buat skema penalti dan blacklist vendor yang terbukti menyajikan makanan basi/busuk.",
            "Publikasikan menu harian dan sumber bahan baku ke kanal resmi sebagai transparansi.",
        ],
    },
    "Higienitas & Sanitasi Dapur": {
        "keywords": [
            "kotor", "lalat", "jorok", "sanitasi", "higienis", "higiene", "jijik",
            "dapur", "kuman", "bakteri", "bersih", "cuci",
        ],
        "finding": (
            "Kata 'dapur' muncul sangat dominan (~1.000x) dalam tweet negatif, sering bersama "
            "indikasi kondisi tidak higienis. Ini menunjukkan publik mempertanyakan "
            "kondisi sanitasi tempat produksi makanan, bukan hanya hasil akhirnya."
        ),
        "recommendations": [
            "Inspeksi rutin Satuan Pelayanan Pemenuhan Gizi (SPPG) / dapur vendor oleh dinas kesehatan.",
            "Wajibkan sertifikasi laik higiene-sanitasi untuk semua dapur penyedia MBG.",
            "Pelatihan food safety berkala untuk juru masak dan staf dapur.",
            "Sediakan kanal publik untuk laporan kondisi dapur (foto/video) dengan respons SLA jelas.",
            "Pasang kamera dan publikasi rutin dokumentasi proses produksi di dapur skala besar.",
        ],
    },
    "Distribusi & Logistik": {
        "keywords": [
            "terlambat", "telat", "antar", "sampai", "lama", "distribusi",
            "logistik", "porsi", "kemasan", "wadah", "dingin", "panas",
        ],
        "finding": (
            "Sebagian tweet mengeluhkan keterlambatan dan kondisi makanan saat sampai. "
            "Makanan yang terlalu lama dalam perjalanan tanpa kontrol suhu adalah faktor "
            "utama pertumbuhan bakteri penyebab keracunan."
        ),
        "recommendations": [
            "Tetapkan jeda waktu maksimum dari masak ke konsumsi (cold chain / hot chain).",
            "Standarisasi wadah penyajian yang menjaga suhu dan mencegah kontaminasi silang.",
            "Petakan ulang radius distribusi tiap dapur agar tidak melebihi kapasitas logistik aman.",
            "Catat dan publikasikan waktu masak vs waktu distribusi sebagai bagian laporan harian.",
            "Sediakan armada distribusi berpendingin untuk wilayah dengan jarak tempuh panjang.",
        ],
    },
    "Pengawasan & Mutu Vendor": {
        "keywords": [
            "vendor", "catering", "pengawasan", "audit", "standar", "kontrol",
            "supplier", "penyedia", "awasi", "pengawas", "sppg",
        ],
        "finding": (
            "Publik menuntut pengawasan vendor yang lebih ketat. Banyak narasi yang "
            "mempertanyakan proses seleksi penyedia dan mekanisme audit yang ada."
        ),
        "recommendations": [
            "Perketat proses lelang/seleksi vendor: prasyarat sertifikasi gizi dan track record.",
            "Bangun sistem rating vendor publik berdasar audit dan laporan insiden.",
            "Audit mendadak (mystery audit) minimum 2x per semester per vendor.",
            "Wajibkan vendor menyertakan ahli gizi dan food safety officer bersertifikat.",
            "Buat sistem evaluasi kinerja vendor berbasis indikator terukur (KPI) dengan publikasi periodik.",
        ],
    },
    "Respons & Penanganan Pemerintah": {
        "keywords": [
            "tanggap", "lambat", "abai", "pertanggungjawaban", "klarifikasi",
            "respons", "tanggung", "jawab", "peduli", "diam", "abaikan",
            "evaluasi", "tutup",
        ],
        "finding": (
            "Banyak tweet mengkritik lambatnya respons dan kurangnya pertanggungjawaban "
            "saat insiden terjadi. Frasa 'tutup mata', 'tutup telinga', dan tuntutan evaluasi "
            "muncul cukup sering."
        ),
        "recommendations": [
            "Bangun SOP respons insiden keracunan dengan SLA terukur (mis. konferensi pers <24 jam).",
            "Sediakan kanal pengaduan publik resmi (hotline / portal) dengan tracking status.",
            "Publikasikan laporan investigasi setiap insiden secara terbuka.",
            "Bentuk tim tanggap darurat lintas instansi (BGN, BPOM, Kemenkes, Pemda).",
            "Lakukan evaluasi program berkala dengan melibatkan publik dan akademisi.",
        ],
    },
    "Dampak Kesehatan": {
        "keywords": [
            "diare", "muntah", "sakit", "dirawat", "korban", "rumah", "sakit",
            "opname", "gejala", "mual", "pusing", "demam", "rs", "puskesmas",
            "klenger", "perut",
        ],
        "finding": (
            "Banyak laporan dampak kesehatan: muntah, diare, hingga rawat inap. "
            "Kata 'korban' muncul ~2.400x, menandakan publik melihat kasus ini bukan "
            "insiden terisolasi melainkan pola berulang."
        ),
        "recommendations": [
            "Wajibkan pelaporan insiden ke dinas kesehatan dalam 1x24 jam dengan format standar.",
            "Bangun dashboard nasional pemantauan insiden keracunan MBG yang transparan.",
            "Sediakan asuransi/jaminan biaya kesehatan untuk korban keracunan MBG.",
            "Koordinasi dengan puskesmas dan RSUD untuk respon cepat dan investigasi epidemiologis.",
            "Lakukan surveilans aktif: cek sampel makanan dan kondisi siswa secara berkala.",
        ],
    },
    "Akuntabilitas & Anggaran": {
        "keywords": [
            "anggaran", "apbn", "korupsi", "transparansi", "dana", "biaya",
            "miliar", "triliun", "hemat", "pejabat", "rupiah", "duit", "uang",
        ],
        "finding": (
            "Sebagian besar kritik mengaitkan kualitas rendah dengan dugaan masalah anggaran: "
            "harga per porsi terlalu rendah, alokasi tidak transparan, hingga indikasi penyalahgunaan."
        ),
        "recommendations": [
            "Publikasikan rincian harga satuan per porsi per daerah secara terbuka.",
            "Audit independen oleh BPK terhadap penggunaan anggaran MBG, hasil dipublikasikan.",
            "Tinjau ulang harga per porsi agar realistis dengan standar gizi yang ditargetkan.",
            "Buka data alokasi anggaran per provinsi/kabupaten secara real-time di portal resmi.",
            "Libatkan masyarakat sipil dan akademisi dalam pengawasan anggaran (social audit).",
        ],
    },
}

# Color palette per aspek (konsisten di semua chart)
ASPECT_COLORS = {
    "Kualitas Makanan & Bahan Baku": "#d62728",
    "Higienitas & Sanitasi Dapur": "#ff7f0e",
    "Distribusi & Logistik": "#2ca02c",
    "Pengawasan & Mutu Vendor": "#1f77b4",
    "Respons & Penanganan Pemerintah": "#9467bd",
    "Dampak Kesehatan": "#e377c2",
    "Akuntabilitas & Anggaran": "#8c564b",
}


# =====================================================================
# DATA LOADING & CACHING
# =====================================================================
@st.cache_data(show_spinner="Memuat data sentimen...")
def load_sentiment_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)

    def parse_tokens(s):
        if not isinstance(s, str):
            return []
        try:
            return ast.literal_eval(s)
        except Exception:
            return []

    df["tokens"] = df["stopword_removal"].apply(parse_tokens)
    df["token_set"] = df["tokens"].apply(set)
    df["text_clean"] = df["tokens"].apply(lambda t: " ".join(t))
    return df


@st.cache_data(show_spinner="Mengategorikan tweet ke aspek...")
def categorize_aspects(df: pd.DataFrame) -> pd.DataFrame:
    aspect_keywords = {name: set(spec["keywords"]) for name, spec in ASPECTS.items()}

    def get_aspects(token_set):
        return [name for name, kws in aspect_keywords.items() if token_set & kws]

    df = df.copy()
    df["aspects"] = df["token_set"].apply(get_aspects)
    df["n_aspects"] = df["aspects"].apply(len)
    return df


@st.cache_data
def load_topic_data():
    df_topics = pd.read_csv(TOPIC_ASSIGN_PATH)
    df_meta = pd.read_csv(TOPIC_META_PATH)
    df_2d = pd.read_csv(TOPIC_2D_PATH)
    return df_topics, df_meta, df_2d


@st.cache_data
def compute_aspect_counts(df_neg: pd.DataFrame) -> pd.DataFrame:
    counter = Counter()
    for asp_list in df_neg["aspects"]:
        for a in asp_list:
            counter[a] += 1
    rows = [
        {
            "aspek": a,
            "jumlah_tweet": counter.get(a, 0),
            "persen_dari_negatif": counter.get(a, 0) / max(len(df_neg), 1) * 100,
        }
        for a in ASPECTS.keys()
    ]
    return pd.DataFrame(rows).sort_values("jumlah_tweet", ascending=False)


@st.cache_data
def compute_top_keywords(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    noise = {
        "mbg", "keracunan", "com", "kali", "name", "format", "tuh", "kok",
        "lah", "sih", "deh", "yah", "lho", "kan", "udah", "aja", "gitu",
        "gini", "nya",
    }
    counter = Counter()
    for tokens in df["tokens"]:
        for t in tokens:
            if t and t not in noise and len(t) > 2:
                counter[t] += 1
    items = counter.most_common(top_n)
    return pd.DataFrame(items, columns=["kata", "frekuensi"])


@st.cache_data
def compute_topic_aspect_overlap(df_topics: pd.DataFrame, df_neg_categorized: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-tab: untuk tiap topik AI, hitung distribusi aspek manual.
    Join lewat full_text (asumsi unique enough).
    """
    df_join = df_topics.merge(
        df_neg_categorized[["full_text", "aspects"]],
        on="full_text",
        how="inner",
    )
    rows = []
    for tid in df_join["dominant_topic"].unique():
        subset = df_join[df_join["dominant_topic"] == tid]
        aspect_counter = Counter()
        for asp_list in subset["aspects"]:
            for a in asp_list:
                aspect_counter[a] += 1
        if not aspect_counter:
            continue
        for asp in ASPECTS.keys():
            rows.append({
                "topic_id": tid,
                "aspek": asp,
                "jumlah": aspect_counter.get(asp, 0),
                "persen_di_topik": aspect_counter.get(asp, 0) / len(subset) * 100,
            })
    return pd.DataFrame(rows)


# =====================================================================
# SIDEBAR
# =====================================================================
with st.sidebar:
    st.markdown("### Filter & Pengaturan")

    conf_threshold = st.slider(
        "Confidence minimum (IndoBERT)",
        min_value=0.0,
        max_value=1.0,
        value=0.70,
        step=0.05,
        help=(
            "Hanya tweet dengan confidence ≥ threshold yang dianalisis. "
            "Threshold lebih tinggi = analisis lebih kredibel tapi data lebih sedikit."
        ),
    )

    st.markdown("---")
    st.markdown("### Tentang Pipeline AI")
    st.markdown(
        "**Layer 1 — Klasifikasi Sentimen**  \n"
        "IndoBERT (transformer fine-tuned) + TF-IDF/SVM → label final (negative / neutral / positive / uncertain)."
    )
    st.markdown(
        "**Layer 2 — Aspect Mapping (Rule-based)**  \n"
        "7 aspek operasional MBG, multi-label keyword matching. Berperan sebagai *interpretable baseline*."
    )
    st.markdown(
        "**Layer 3 — Topic Discovery (NMF)**  \n"
        "Non-negative Matrix Factorization pada TF-IDF tweet negatif → 6 topik tematik *unsupervised*. "
        "Menemukan struktur tematik yang tidak terlihat dari rule-based."
    )

    st.markdown("---")
    st.caption("Tugas Besar Kecerdasan Artifisial & Penerapannya")
    st.caption("Telkom University Purwokerto")


# =====================================================================
# LOAD DATA
# =====================================================================
df = load_sentiment_data()
df = df[df["confidence_indobert"] >= conf_threshold].copy()
df = categorize_aspects(df)

df_topic_assign, df_topic_meta, df_topic_2d = load_topic_data()


# =====================================================================
# HEADER
# =====================================================================
st.title("Dashboard Evaluasi BGN")
st.markdown("##### Analisis Sentimen Keracunan MBG di Platform X — Pipeline AI Tiga-Layer")
st.markdown(
    "Dashboard ini menerjemahkan hasil model AI (IndoBERT + SVM) "
    "menjadi **insight evaluasi konkret** untuk Badan Gizi Nasional (BGN). "
    "Menjawab pertanyaan: _bagian mana dari program MBG yang perlu dievaluasi, "
    "dan tema diskursus apa yang muncul secara data-driven dari publik?_"
)

# Methodology callout
with st.expander("🔬 Lihat pipeline AI yang digunakan", expanded=False):
    st.markdown(
        """
**Layer 1: Klasifikasi Sentimen (Supervised, sudah ada)**  
Tweet → preprocessing (cleaning, casefolding, normalisasi, stemming) → 
**IndoBERT** (transformer) untuk semantic understanding + **TF-IDF + SVM** untuk klasifikasi → 
label final (negative / neutral / positive / uncertain).

**Layer 2: Aspect Mapping (Rule-based, interpretable baseline)**  
Tweet negatif → matching ke 7 aspek operasional MBG via dictionary keyword → 
multi-label (1 tweet bisa masuk beberapa aspek). Dipilih karena deterministik dan defensible.

**Layer 3: Topic Discovery (NMF Unsupervised — BARU di revisi)**  
TF-IDF vectorization (max_features=2500, max_df=0.35) → 
**Non-negative Matrix Factorization** (6 components) → 
6 topik tematik discovered tanpa supervisi → 
**menemukan insight yang tidak terlihat dari rule-based** (mis. plesetan 'Makan Beracun Gratis', 
frekuensi berita harian, persepsi statistik korban).

**Layer 4: Konvergensi**  
Cross-validation hasil Layer 2 (rule-based) vs Layer 3 (NMF) → mengukur konsistensi dua pendekatan.
        """
    )

st.markdown("---")


# =====================================================================
# SECTION 1: OVERVIEW
# =====================================================================
st.header("1. Ringkasan Distribusi Sentimen")

label_counts = df["label_final"].value_counts()
total = len(df)
n_neg = int(label_counts.get("negative", 0))
n_neu = int(label_counts.get("neutral", 0))
n_pos = int(label_counts.get("positive", 0))
n_unc = int(label_counts.get("uncertain", 0))

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Tweet", f"{total:,}")
c2.metric("Negatif", f"{n_neg:,}", f"{n_neg/max(total,1)*100:.1f}%")
c3.metric("Netral", f"{n_neu:,}", f"{n_neu/max(total,1)*100:.1f}%")
c4.metric("Positif", f"{n_pos:,}", f"{n_pos/max(total,1)*100:.1f}%")
c5.metric("Uncertain", f"{n_unc:,}", f"{n_unc/max(total,1)*100:.1f}%")

dist_df = pd.DataFrame({
    "Sentimen": ["Negatif", "Netral", "Positif", "Uncertain"],
    "Jumlah": [n_neg, n_neu, n_pos, n_unc],
})
fig_dist = px.bar(
    dist_df,
    x="Sentimen",
    y="Jumlah",
    color="Sentimen",
    color_discrete_map={
        "Negatif": "#d62728",
        "Netral": "#7f7f7f",
        "Positif": "#2ca02c",
        "Uncertain": "#ff7f0e",
    },
    text="Jumlah",
)
fig_dist.update_traces(textposition="outside")
fig_dist.update_layout(showlegend=False, yaxis_title="Jumlah tweet", xaxis_title="")
st.plotly_chart(fig_dist, use_container_width=True)

st.info(
    f"**Temuan**: dari {total:,} tweet yang dianalisis IndoBERT + SVM, "
    f"**{n_neg/max(total,1)*100:.1f}% negatif**, hanya {n_pos/max(total,1)*100:.1f}% positif. "
    f"Imbalance ekstrem ini sendiri sudah temuan: publik di X sangat dominan menyuarakan kritik "
    f"terhadap MBG terkait isu keracunan."
)


# =====================================================================
# SECTION 2: ASPECT-BASED (RULE-BASED, INTERPRETABLE BASELINE)
# =====================================================================
st.markdown("---")
st.header("2. Layer 2 — Aspect Mapping Rule-based")
st.markdown(
    "**Pendekatan**: tweet negatif dimapping ke **7 aspek operasional MBG** via keyword dictionary matching. "
    "Multi-label (satu tweet bisa masuk beberapa aspek). Berperan sebagai *interpretable baseline* "
    "yang setiap kategorisasinya bisa ditelusuri ke keyword spesifik."
)

df_neg = df[df["label_final"] == "negative"].copy()
n_uncategorized = (df_neg["n_aspects"] == 0).sum()
aspect_counts = compute_aspect_counts(df_neg)

colA, colB = st.columns([2, 1])
with colA:
    fig_asp = px.bar(
        aspect_counts,
        x="jumlah_tweet",
        y="aspek",
        orientation="h",
        text="jumlah_tweet",
        color="aspek",
        color_discrete_map=ASPECT_COLORS,
    )
    fig_asp.update_traces(textposition="outside")
    fig_asp.update_layout(
        yaxis={"categoryorder": "total ascending"},
        xaxis_title="Jumlah tweet negatif",
        yaxis_title="",
        showlegend=False,
        height=420,
    )
    st.plotly_chart(fig_asp, use_container_width=True)

with colB:
    st.metric("Total tweet negatif", f"{len(df_neg):,}")
    cat_rate = (len(df_neg) - n_uncategorized) / max(len(df_neg), 1) * 100
    st.metric(
        "Tweet ter-kategorisasi",
        f"{len(df_neg) - n_uncategorized:,}",
        f"{cat_rate:.1f}%",
    )
    st.caption(
        f"{n_uncategorized:,} tweet tidak masuk aspek mana pun (kritik umum tanpa "
        "kata kunci aspek spesifik). Total per aspek > total tweet karena multi-label."
    )

st.subheader("Top 20 kata kunci di tweet negatif")
top_kw = compute_top_keywords(df_neg, top_n=20)
fig_kw = px.bar(
    top_kw,
    x="frekuensi",
    y="kata",
    orientation="h",
    text="frekuensi",
    color="frekuensi",
    color_continuous_scale="Reds",
)
fig_kw.update_traces(textposition="outside")
fig_kw.update_layout(
    yaxis={"categoryorder": "total ascending"},
    xaxis_title="Frekuensi",
    yaxis_title="",
    coloraxis_showscale=False,
    height=500,
)
st.plotly_chart(fig_kw, use_container_width=True)


# =====================================================================
# SECTION 3: NMF TOPIC DISCOVERY (UNSUPERVISED — INSIGHT BARU)
# =====================================================================
st.markdown("---")
st.header("3. Layer 3 — Topic Discovery via NMF (Unsupervised AI)")
st.markdown(
    "**Pendekatan**: **Non-negative Matrix Factorization** pada TF-IDF tweet negatif "
    "menemukan **6 topik tematik secara data-driven**, tanpa pre-defined keywords. "
    "Tujuan: mengungkap tema diskursus yang **tidak terlihat dari rule-based**."
)

# Topic distribution chart
df_topic_meta_display = df_topic_meta.copy()
fig_topic = px.bar(
    df_topic_meta_display,
    x="size",
    y="suggested_name",
    orientation="h",
    text="size",
    color="suggested_name",
    color_discrete_sequence=px.colors.qualitative.Set2,
    hover_data={"top_terms": True, "share_pct": True, "size": False, "suggested_name": False},
)
fig_topic.update_traces(textposition="outside")
fig_topic.update_layout(
    yaxis={"categoryorder": "total ascending"},
    xaxis_title="Jumlah tweet (berdasar dominant topic)",
    yaxis_title="",
    showlegend=False,
    height=400,
)
st.plotly_chart(fig_topic, use_container_width=True)

# Scatter plot 2D
st.subheader("Visualisasi 2D struktur topik (TruncatedSVD)")
st.markdown(
    "Sample 5.000 tweet, diproyeksikan ke 2D pakai TruncatedSVD dari ruang TF-IDF. "
    "Tweet yang berdekatan di plot = punya kata kunci yang mirip."
)

# Map topic_id ke nama untuk display
topic_id_to_name = dict(zip(df_topic_meta["topic_id"], df_topic_meta["suggested_name"]))
df_topic_2d_display = df_topic_2d.copy()
df_topic_2d_display["topic_name"] = df_topic_2d_display["topic_id"].map(topic_id_to_name)
df_topic_2d_display["preview"] = df_topic_2d_display["full_text"].astype(str).str[:120] + "..."

fig_2d = px.scatter(
    df_topic_2d_display,
    x="x",
    y="y",
    color="topic_name",
    color_discrete_sequence=px.colors.qualitative.Set2,
    hover_data={"preview": True, "x": False, "y": False, "topic_name": True},
    opacity=0.5,
    height=500,
)
fig_2d.update_traces(marker=dict(size=4))
fig_2d.update_layout(
    xaxis_title="Komponen 1",
    yaxis_title="Komponen 2",
    legend_title="Topik",
)
st.plotly_chart(fig_2d, use_container_width=True)

# Topic detail expanders
st.subheader("Detail per topik")
st.markdown(
    "Klik untuk lihat *top terms* dan **interpretasi** tiap topik — termasuk topik yang "
    "merupakan insight baru tidak ditemukan rule-based."
)

for _, row in df_topic_meta.iterrows():
    label = f"**T{row['topic_id']} — {row['suggested_name']}** ({row['size']:,} tweet, {row['share_pct']:.1f}%)"
    with st.expander(label, expanded=False):
        st.markdown("**Top terms (data-driven, dari NMF):**")
        st.markdown(", ".join([f"`{t.strip()}`" for t in row["top_terms"].split(",")]))
        st.markdown("**Mapping ke aspek/dimensi evaluasi:**")
        st.markdown(f"_{row['mapped_aspect']}_")
        st.markdown("**Interpretasi & insight:**")
        st.markdown(f"> {row['interpretation']}")


# =====================================================================
# SECTION 4: KONVERGENSI ASPEK ↔ TOPIK (CROSS-VALIDATION)
# =====================================================================
st.markdown("---")
st.header("4. Layer 4 — Konvergensi Aspek (Rule-based) ↔ Topik (NMF)")
st.markdown(
    "**Pendekatan**: cross-validation antara kategori manual (Layer 2) dan topik AI (Layer 3). "
    "Untuk tiap topik NMF, lihat distribusi aspek manual di dalamnya. "
    "Konvergensi tinggi = dua pendekatan sepakat → rekomendasi BGN di aspek itu lebih kredibel."
)

with st.spinner("Computing cross-validation overlap..."):
    overlap = compute_topic_aspect_overlap(df_topic_assign, df_neg)

if len(overlap) > 0:
    # Pivot untuk heatmap
    overlap = overlap.merge(
        df_topic_meta[["topic_id", "suggested_name"]],
        on="topic_id",
        how="left",
    )
    overlap["topik"] = "T" + overlap["topic_id"].astype(str) + " — " + overlap["suggested_name"]

    pivot = overlap.pivot(index="topik", columns="aspek", values="persen_di_topik").fillna(0)

    fig_heat = px.imshow(
        pivot,
        text_auto=".1f",
        aspect="auto",
        color_continuous_scale="Reds",
        labels=dict(x="Aspek (Rule-based)", y="Topik (NMF)", color="% di topik"),
    )
    fig_heat.update_layout(height=420)
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown(
        "**Cara membaca**: angka di tiap sel = % tweet di topik tertentu yang juga ter-tag "
        "ke aspek manual tertentu. Sel merah pekat = konvergensi kuat antara dua pendekatan."
    )

    st.info(
        "**Insight dari konvergensi**:\n"
        "- Topik **'Kualitas Makanan Basi & Dapur'** (NMF) konvergen kuat dengan aspek "
        "**Kualitas Makanan** & **Higienitas Dapur** (rule-based) → validasi data-driven "
        "untuk rekomendasi BGN di dua aspek ini.\n"
        "- Topik **'Plesetan Makan Beracun Gratis'** dan **'Frekuensi Berita Harian'** "
        "adalah dimensi yang **tidak ter-tag aspek manual mana pun** → insight baru: "
        "publik bukan cuma kritik operasional, tapi juga kritik **reputasi program** "
        "dan **persepsi krisis berkelanjutan**.\n"
        "- BGN perlu strategi **komunikasi krisis** dan **rebranding**, bukan hanya "
        "perbaikan operasional dapur/vendor."
    )
else:
    st.warning("Cross-validation tidak bisa dihitung (join gagal).")


# =====================================================================
# SECTION 5: REKOMENDASI EVALUASI BGN
# =====================================================================
st.markdown("---")
st.header("5. Rekomendasi Evaluasi untuk BGN")
st.markdown(
    "Setiap aspek diberi **temuan ringkas** (berbasis data tweet) dan **rekomendasi aksi konkret**. "
    "Diurutkan dari aspek yang paling banyak dikeluhkan publik."
)

sorted_aspects = aspect_counts.set_index("aspek").index.tolist()

for asp_name in sorted_aspects:
    spec = ASPECTS[asp_name]
    n = int(aspect_counts.loc[aspect_counts["aspek"] == asp_name, "jumlah_tweet"].iloc[0])
    pct = float(aspect_counts.loc[aspect_counts["aspek"] == asp_name, "persen_dari_negatif"].iloc[0])

    with st.expander(f"**{asp_name}** — {n:,} tweet ({pct:.1f}% dari negatif)", expanded=False):
        st.markdown("**Kata kunci yang dideteksi:**")
        st.markdown(", ".join([f"`{k}`" for k in spec["keywords"]]))

        st.markdown("**Temuan ringkas:**")
        st.markdown(f"> {spec['finding']}")

        st.markdown("**Rekomendasi aksi untuk BGN:**")
        for i, rec in enumerate(spec["recommendations"], 1):
            st.markdown(f"{i}. {rec}")

# Rekomendasi tambahan dari NMF (yang tidak ada di aspek manual)
st.subheader("Rekomendasi tambahan dari Topic Discovery (NMF)")
st.markdown(
    "Topik berikut **tidak ditemukan oleh rule-based aspek manual** — murni hasil unsupervised AI:"
)

with st.expander("**Reputasi Program** — Plesetan 'Makan Beracun Gratis' (2.582 tweet)", expanded=False):
    st.markdown("**Temuan dari NMF:**")
    st.markdown(
        "> Publik aktif mem-plesetkan akronim MBG dari 'Makan Bergizi Gratis' menjadi "
        "'Makan Beracun Gratis'. Reputasi program rusak di level branding, bukan hanya operasional."
    )
    st.markdown("**Rekomendasi:**")
    st.markdown(
        "1. Bangun strategi komunikasi positif dengan publikasi success story per daerah.\n"
        "2. Aktifkan crisis communication officer untuk respons cepat insiden.\n"
        "3. Engage influencer pendidikan dan ahli gizi sebagai third-party endorsement.\n"
        "4. Pertimbangkan rebranding atau penambahan tagline baru yang menetralkan plesetan negatif."
    )

with st.expander("**Komunikasi Krisis** — Frekuensi Berita Harian (1.857 tweet)", expanded=False):
    st.markdown("**Temuan dari NMF:**")
    st.markdown(
        "> Publik merasa berita keracunan terjadi 'tiap hari', menciptakan persepsi "
        "krisis berkelanjutan. Bukan kritik teknis, tapi *news fatigue*."
    )
    st.markdown("**Rekomendasi:**")
    st.markdown(
        "1. Publikasikan laporan agregat bulanan: total dapur, total porsi, total insiden "
        "(dengan rasio). Konteks meredakan persepsi 'tiap hari'.\n"
        "2. Counter-narrative dengan data: 'X juta porsi disajikan, Y insiden = Z% rate'.\n"
        "3. Aktif di media sosial dengan content yang substantif, bukan hanya defensif.\n"
        "4. Sediakan press briefing rutin untuk wartawan dan kreator pendidikan."
    )

with st.expander("**Transparansi Data Korban** — Skala Korban Massal (1.803 tweet)", expanded=False):
    st.markdown("**Temuan dari NMF:**")
    st.markdown(
        "> Diskursus berfokus pada jumlah korban ('ribuan jiwa'), bukan akar masalah. "
        "Publik mempertanyakan transparansi data korban resmi."
    )
    st.markdown("**Rekomendasi:**")
    st.markdown(
        "1. Bangun dashboard publik real-time: jumlah insiden, jumlah korban per insiden, status investigasi.\n"
        "2. Definisi 'korban' yang konsisten dan diumumkan publik (kriteria gejala minimal).\n"
        "3. Update mingguan, bukan menunggu wartawan menghitung sendiri.\n"
        "4. Kolaborasi dengan Kemenkes untuk validasi angka resmi."
    )


# =====================================================================
# SECTION 6: EKSPLORASI TWEET
# =====================================================================
st.markdown("---")
st.header("6. Eksplorasi Tweet")

tab1, tab2 = st.tabs(["Per Aspek (Rule-based)", "Per Topik (NMF)"])

with tab1:
    colS1, colS2 = st.columns([2, 1])
    with colS1:
        selected_aspect = st.selectbox(
            "Pilih aspek",
            options=list(ASPECTS.keys()),
            key="aspect_select",
        )
    with colS2:
        n_samples_asp = st.number_input(
            "Jumlah sample",
            min_value=5, max_value=30, value=10, step=5,
            key="n_asp",
        )

    df_aspect = df_neg[df_neg["aspects"].apply(lambda lst: selected_aspect in lst)].copy()
    df_aspect = df_aspect.sort_values("confidence_indobert", ascending=False)

    st.caption(
        f"Menampilkan {min(n_samples_asp, len(df_aspect))} dari {len(df_aspect):,} tweet "
        f"yang masuk aspek **{selected_aspect}** (diurutkan dari confidence tertinggi)."
    )

    for idx, row in df_aspect.head(n_samples_asp).iterrows():
        other_aspects = [a for a in row["aspects"] if a != selected_aspect]
        other_str = f" | Juga masuk: {', '.join(other_aspects)}" if other_aspects else ""
        st.markdown(
            f"- _{row['full_text'][:400]}_  \n"
            f"  <small>Confidence: {row['confidence_indobert']:.2f}{other_str}</small>",
            unsafe_allow_html=True,
        )

with tab2:
    colT1, colT2 = st.columns([2, 1])
    with colT1:
        topic_options = dict(zip(
            df_topic_meta["topic_id"].astype(str) + " — " + df_topic_meta["suggested_name"],
            df_topic_meta["topic_id"]
        ))
        selected_topic_label = st.selectbox(
            "Pilih topik (NMF)",
            options=list(topic_options.keys()),
            key="topic_select",
        )
        selected_topic_id = topic_options[selected_topic_label]
    with colT2:
        n_samples_topic = st.number_input(
            "Jumlah sample",
            min_value=5, max_value=30, value=10, step=5,
            key="n_topic",
        )

    df_topic_subset = df_topic_assign[df_topic_assign["dominant_topic"] == selected_topic_id].copy()
    df_topic_subset = df_topic_subset.sort_values("topic_score", ascending=False)

    st.caption(
        f"Menampilkan {min(n_samples_topic, len(df_topic_subset))} dari "
        f"{len(df_topic_subset):,} tweet yang dominant topic-nya = topik ini "
        f"(diurutkan dari topic strength tertinggi)."
    )

    for _, row in df_topic_subset.head(n_samples_topic).iterrows():
        st.markdown(
            f"- _{str(row['full_text'])[:400]}_  \n"
            f"  <small>Topic score: {row['topic_score']:.3f} | "
            f"Sentiment confidence: {row['confidence_indobert']:.2f}</small>",
            unsafe_allow_html=True,
        )


# =====================================================================
# FOOTER
# =====================================================================
st.markdown("---")
with st.expander("📋 Catatan metodologi, batasan, dan pertimbangan akademik"):
    st.markdown(
        """
### Pipeline 4 layer
1. **IndoBERT + TF-IDF/SVM** (supervised, sudah dilatih di notebook): klasifikasi sentimen 4 kelas.
2. **Rule-based Aspect Mapping**: 7 aspek operasional, multi-label keyword matching — interpretable baseline.
3. **NMF Topic Modeling** (unsupervised): TF-IDF + Non-negative Matrix Factorization (6 components), 
   max_features=2500, max_df=0.35, min_df=20, L2-normalized, sublinear TF.
4. **Konvergensi**: cross-validation aspek-topik via document join.

### Mengapa hybrid?
- **Rule-based** memberikan *interpretability*: tiap kategorisasi traceable ke keyword.
- **NMF unsupervised** memberikan *discovery*: menemukan tema tanpa bias keyword manual.
- **Konvergensi** memberikan *validasi*: dua metode berbeda yang sepakat → temuan kredibel.

### Insight yang ditemukan AI tapi tidak ditemukan rule-based
- **Reputasi program**: plesetan 'Makan Beracun Gratis' (9.4% tweet negatif).
- **Komunikasi krisis**: persepsi 'berita tiap hari' (6.8%).
- **Transparansi data korban**: fokus diskursus pada angka korban (6.6%).

### Batasan
- Tidak ada kolom tanggal pada dataset → tidak ada analisis time series.
- Tweet pendek + vocabulary heterogen → cluster dominan (T0, 63.8%) tidak terhindarkan.
- T4 (5.4%) adalah noise konten media, dikecualikan dari interpretasi.
- Rule-based keyword bersifat indikatif; sarkasme dan bahasa daerah berat bisa salah klasifikasi.

### Reprodusibilitas
- Random seed: 42 (NMF, sampling).
- Semua kategorisasi deterministik. Tidak ada LLM call atau API external.
- Topic assignments dapat dihitung ulang dari dataset asli + parameter di notebook.
        """
    )

st.caption(
    "Dashboard ini adalah revisi tugas besar Kecerdasan Artifisial & Penerapannya: "
    "menerjemahkan output klasifikasi sentimen menjadi insight evaluasi yang dapat "
    "ditindaklanjuti BGN, dengan tambahan unsupervised AI (NMF) untuk discovery insight baru."
)
