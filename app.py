"""
Dashboard Sentimen MBG di X
"""

import ast
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# =====================================================================
# CONFIG
# =====================================================================
st.set_page_config(
    page_title="Sentimen MBG di X",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CSV_PATH = "hasil_sentimen_slim.csv"
TOPIC_ASSIGN_PATH = "topic_assignments.csv"
TOPIC_META_PATH = "topic_metadata.csv"
TOPIC_2D_PATH = "topic_2d_sample.csv"

COLOR_NEG = "#DC2626"
COLOR_NEU = "#6B7280"
COLOR_POS = "#10B981"
COLOR_UNC = "#F59E0B"

# =====================================================================
# CUSTOM CSS
# =====================================================================
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    .hero-title {
        font-size: 2.25rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem;
        line-height: 1.1;
    }
    .hero-sub {
        font-size: 1rem;
        opacity: 0.7;
        margin-bottom: 1.5rem;
    }

    [data-testid="stMetric"] {
        background: rgba(127, 127, 127, 0.06);
        padding: 1rem 1.25rem;
        border-radius: 10px;
        border: 1px solid rgba(127, 127, 127, 0.12);
    }
    [data-testid="stMetricValue"] {
        font-weight: 700;
        font-size: 1.75rem;
    }
    [data-testid="stMetricLabel"] {
        font-weight: 500;
        opacity: 0.8;
        font-size: 0.875rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.25rem;
        border-bottom: 1px solid rgba(127, 127, 127, 0.15);
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        padding: 0 1.25rem;
        font-weight: 500;
        font-size: 0.95rem;
    }
    .stTabs [aria-selected="true"] {
        font-weight: 600;
    }

    .section-title {
        font-size: 1.15rem;
        font-weight: 600;
        margin: 1rem 0 0.5rem 0;
    }
    .section-sub {
        font-size: 0.85rem;
        opacity: 0.65;
        margin-bottom: 1rem;
    }

    .insight-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin: 1rem 0 1.5rem 0;
    }
    .insight-card {
        background: linear-gradient(135deg, rgba(220, 38, 38, 0.07), rgba(220, 38, 38, 0.02));
        border-left: 3px solid #DC2626;
        padding: 1.1rem 1.25rem;
        border-radius: 8px;
    }
    .insight-tag {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        color: #DC2626;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }
    .insight-title {
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.4rem;
        line-height: 1.3;
    }
    .insight-count {
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }
    .insight-desc {
        font-size: 0.85rem;
        opacity: 0.8;
        line-height: 1.5;
    }

    .tweet-box {
        background: rgba(127, 127, 127, 0.05);
        border-left: 2px solid rgba(127, 127, 127, 0.3);
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        border-radius: 4px;
        font-size: 0.9rem;
    }
    .tweet-meta {
        font-size: 0.75rem;
        opacity: 0.6;
        margin-top: 0.3rem;
    }

    .streamlit-expanderHeader {
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# DATA CONSTANTS
# =====================================================================
ASPECTS = {
    "Kualitas Makanan & Bahan Baku": {
        "keywords": ["basi", "busuk", "mentah", "hambar", "bau", "expired", "kedaluwarsa",
                     "kualitas", "mutu", "ayam", "ikan", "daging", "sayur", "telur",
                     "matang", "asin", "amis", "rasa", "jelek", "murah"],
        "recommendations": [
            "Tetapkan standar mutu bahan baku (HACCP/SNI) wajib dipenuhi vendor sebelum kontrak.",
            "Uji organoleptik dan pengukuran suhu makanan saat tiba di sekolah.",
            "Audit acak pemasok bahan baku minimal 1x/bulan oleh tim independen.",
            "Skema penalti dan blacklist vendor yang menyajikan makanan basi/busuk.",
            "Publikasikan menu harian dan sumber bahan baku ke kanal resmi.",
        ],
    },
    "Higienitas & Sanitasi Dapur": {
        "keywords": ["kotor", "lalat", "jorok", "sanitasi", "higienis", "higiene", "jijik",
                     "dapur", "kuman", "bakteri", "bersih", "cuci"],
        "recommendations": [
            "Inspeksi rutin SPPG/dapur vendor oleh dinas kesehatan setempat.",
            "Wajibkan sertifikasi laik higiene-sanitasi untuk semua dapur penyedia MBG.",
            "Pelatihan food safety berkala untuk juru masak dan staf dapur.",
            "Kanal publik untuk laporan kondisi dapur dengan respons SLA jelas.",
            "Dokumentasi proses produksi terbuka untuk dapur skala besar.",
        ],
    },
    "Distribusi & Logistik": {
        "keywords": ["terlambat", "telat", "antar", "sampai", "lama", "distribusi",
                     "logistik", "porsi", "kemasan", "wadah", "dingin", "panas"],
        "recommendations": [
            "Tetapkan jeda waktu maksimum dari masak ke konsumsi (cold/hot chain).",
            "Standarisasi wadah penyajian penjaga suhu dan anti kontaminasi silang.",
            "Petakan ulang radius distribusi per dapur agar tidak overcapacity.",
            "Catat dan publikasikan waktu masak vs distribusi sebagai laporan harian.",
            "Armada distribusi berpendingin untuk wilayah jarak tempuh panjang.",
        ],
    },
    "Pengawasan & Mutu Vendor": {
        "keywords": ["vendor", "catering", "pengawasan", "audit", "standar", "kontrol",
                     "supplier", "penyedia", "awasi", "pengawas", "sppg"],
        "recommendations": [
            "Perketat seleksi vendor: prasyarat sertifikasi gizi dan track record.",
            "Sistem rating vendor publik berdasar audit dan laporan insiden.",
            "Audit mendadak (mystery audit) minimum 2x/semester per vendor.",
            "Vendor wajib menyertakan ahli gizi dan food safety officer bersertifikat.",
            "Evaluasi kinerja vendor berbasis KPI dengan publikasi periodik.",
        ],
    },
    "Respons & Penanganan Pemerintah": {
        "keywords": ["tanggap", "lambat", "abai", "pertanggungjawaban", "klarifikasi",
                     "respons", "tanggung", "jawab", "peduli", "diam", "abaikan",
                     "evaluasi", "tutup"],
        "recommendations": [
            "SOP respons insiden keracunan dengan SLA terukur (mis. konferensi pers <24 jam).",
            "Kanal pengaduan publik resmi (hotline/portal) dengan tracking status.",
            "Publikasikan laporan investigasi setiap insiden secara terbuka.",
            "Tim tanggap darurat lintas instansi (BGN, BPOM, Kemenkes, Pemda).",
            "Evaluasi program berkala melibatkan publik dan akademisi.",
        ],
    },
    "Dampak Kesehatan": {
        "keywords": ["diare", "muntah", "sakit", "dirawat", "korban", "rumah", "sakit",
                     "opname", "gejala", "mual", "pusing", "demam", "rs", "puskesmas",
                     "klenger", "perut"],
        "recommendations": [
            "Wajibkan pelaporan insiden ke dinas kesehatan dalam 1x24 jam (format standar).",
            "Dashboard nasional pemantauan insiden keracunan MBG yang transparan.",
            "Asuransi/jaminan biaya kesehatan untuk korban keracunan MBG.",
            "Koordinasi puskesmas dan RSUD untuk respon cepat dan investigasi epidemiologis.",
            "Surveilans aktif: cek sampel makanan dan kondisi siswa secara berkala.",
        ],
    },
    "Akuntabilitas & Anggaran": {
        "keywords": ["anggaran", "apbn", "korupsi", "transparansi", "dana", "biaya",
                     "miliar", "triliun", "hemat", "pejabat", "rupiah", "duit", "uang"],
        "recommendations": [
            "Publikasikan rincian harga satuan per porsi per daerah.",
            "Audit independen BPK terhadap penggunaan anggaran MBG, hasil dipublikasikan.",
            "Tinjau ulang harga per porsi agar realistis dengan standar gizi yang ditargetkan.",
            "Buka data alokasi anggaran per provinsi/kabupaten real-time.",
            "Libatkan masyarakat sipil dan akademisi dalam pengawasan anggaran.",
        ],
    },
}

ASPECT_COLORS = {
    "Kualitas Makanan & Bahan Baku": "#DC2626",
    "Higienitas & Sanitasi Dapur": "#F97316",
    "Distribusi & Logistik": "#10B981",
    "Pengawasan & Mutu Vendor": "#3B82F6",
    "Respons & Penanganan Pemerintah": "#8B5CF6",
    "Dampak Kesehatan": "#EC4899",
    "Akuntabilitas & Anggaran": "#A16207",
}

# =====================================================================
# CACHED LOADERS
# =====================================================================
@st.cache_data(show_spinner=False)
def load_sentiment_data():
    df = pd.read_csv(CSV_PATH)
    def parse_tokens(s):
        if not isinstance(s, str): return []
        try: return ast.literal_eval(s)
        except: return []
    df["tokens"] = df["stopword_removal"].apply(parse_tokens)
    df["token_set"] = df["tokens"].apply(set)
    return df

@st.cache_data(show_spinner=False)
def categorize_aspects(df):
    aspect_kw = {n: set(s["keywords"]) for n, s in ASPECTS.items()}
    df = df.copy()
    df["aspects"] = df["token_set"].apply(lambda ts: [n for n, kw in aspect_kw.items() if ts & kw])
    df["n_aspects"] = df["aspects"].apply(len)
    return df

@st.cache_data(show_spinner=False)
def load_topic_data():
    return (
        pd.read_csv(TOPIC_ASSIGN_PATH),
        pd.read_csv(TOPIC_META_PATH),
        pd.read_csv(TOPIC_2D_PATH),
    )

@st.cache_data(show_spinner=False)
def compute_aspect_counts(df_neg):
    counter = Counter()
    for asp_list in df_neg["aspects"]:
        for a in asp_list:
            counter[a] += 1
    rows = [
        {"aspek": a, "jumlah": counter.get(a, 0),
         "persen": counter.get(a, 0) / max(len(df_neg), 1) * 100}
        for a in ASPECTS.keys()
    ]
    return pd.DataFrame(rows).sort_values("jumlah", ascending=False)

@st.cache_data(show_spinner=False)
def compute_top_keywords(df, top_n=20):
    noise = {"mbg", "keracunan", "com", "kali", "name", "format", "tuh", "kok",
             "lah", "sih", "deh", "yah", "lho", "kan", "udah", "aja", "gitu",
             "gini", "nya"}
    counter = Counter()
    for tokens in df["tokens"]:
        for t in tokens:
            if t and t not in noise and len(t) > 2:
                counter[t] += 1
    return pd.DataFrame(counter.most_common(top_n), columns=["kata", "frekuensi"])

@st.cache_data(show_spinner=False)
def compute_overlap(df_topics, df_neg):
    df_join = df_topics.merge(df_neg[["full_text", "aspects"]], on="full_text", how="inner")
    rows = []
    for tid in df_join["dominant_topic"].unique():
        subset = df_join[df_join["dominant_topic"] == tid]
        c = Counter()
        for asp_list in subset["aspects"]:
            for a in asp_list: c[a] += 1
        for asp in ASPECTS.keys():
            rows.append({"topic_id": tid, "aspek": asp,
                         "persen": c.get(asp, 0) / max(len(subset), 1) * 100})
    return pd.DataFrame(rows)


# =====================================================================
# SIDEBAR
# =====================================================================
with st.sidebar:
    st.markdown("### Filter")
    conf_threshold = st.slider(
        "Confidence minimum",
        0.0, 1.0, 0.70, 0.05,
        help="Minimum confidence skor IndoBERT.",
    )
    st.markdown("---")
    st.caption("Tugas Besar Kecerdasan Artifisial & Penerapannya")
    st.caption("Telkom University Purwokerto")


# =====================================================================
# DATA
# =====================================================================
df = load_sentiment_data()
df = df[df["confidence_indobert"] >= conf_threshold].copy()
df = categorize_aspects(df)
df_topic_assign, df_topic_meta, df_topic_2d = load_topic_data()
df_neg = df[df["label_final"] == "negative"].copy()
aspect_counts = compute_aspect_counts(df_neg)


# =====================================================================
# HERO
# =====================================================================
st.markdown('<div class="hero-title">Sentimen MBG di Platform X</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Analisis sentimen publik terhadap insiden keracunan program '
    'Makan Bergizi Gratis dan rekomendasi evaluasi untuk Badan Gizi Nasional.</div>',
    unsafe_allow_html=True,
)


# =====================================================================
# KPI ROW
# =====================================================================
total = len(df)
n_neg = int((df["label_final"] == "negative").sum())
n_neu = int((df["label_final"] == "neutral").sum())
n_pos = int((df["label_final"] == "positive").sum())
n_unc = int((df["label_final"] == "uncertain").sum())

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Tweet", f"{total:,}")
c2.metric("Negatif", f"{n_neg:,}", f"{n_neg/max(total,1)*100:.1f}%")
c3.metric("Netral", f"{n_neu:,}", f"{n_neu/max(total,1)*100:.1f}%")
c4.metric("Positif", f"{n_pos:,}", f"{n_pos/max(total,1)*100:.1f}%")
c5.metric("Uncertain", f"{n_unc:,}", f"{n_unc/max(total,1)*100:.1f}%")

st.markdown("")


# =====================================================================
# TABS
# =====================================================================
tab1, tab2, tab3, tab4 = st.tabs(["Aspek Kritik", "Topik AI", "Rekomendasi", "Eksplorasi"])


# ---------------------------------------------------------------------
# TAB 1: ASPEK
# ---------------------------------------------------------------------
with tab1:
    colL, colR = st.columns([3, 2])

    with colL:
        st.markdown('<div class="section-title">7 Aspek Operasional di Tweet Negatif</div>',
                    unsafe_allow_html=True)
        fig_asp = px.bar(
            aspect_counts, x="jumlah", y="aspek", orientation="h",
            text="jumlah", color="aspek", color_discrete_map=ASPECT_COLORS,
        )
        fig_asp.update_traces(textposition="outside", textfont_size=11)
        fig_asp.update_layout(
            yaxis={"categoryorder": "total ascending"},
            xaxis_title="", yaxis_title="",
            showlegend=False, height=380,
            margin=dict(t=20, b=20, l=10, r=40),
            plot_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_asp, use_container_width=True)

    with colR:
        st.markdown('<div class="section-title">Top 15 Kata Kunci</div>',
                    unsafe_allow_html=True)
        top_kw = compute_top_keywords(df_neg, top_n=15)
        fig_kw = px.bar(
            top_kw, x="frekuensi", y="kata", orientation="h",
            color="frekuensi", color_continuous_scale=["#FCA5A5", "#DC2626"],
        )
        fig_kw.update_layout(
            yaxis={"categoryorder": "total ascending"},
            xaxis_title="", yaxis_title="",
            coloraxis_showscale=False, height=380,
            margin=dict(t=20, b=20, l=10, r=20),
            plot_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_kw, use_container_width=True)


# ---------------------------------------------------------------------
# TAB 2: TOPIK AI
# ---------------------------------------------------------------------
with tab2:
    st.markdown('<div class="section-title">3 Insight Unik dari Model AI</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Temuan dari unsupervised NMF yang tidak terdeteksi '
        'keyword aspek manual.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("""
    <div class="insight-grid">
        <div class="insight-card">
            <div class="insight-tag">REPUTASI PROGRAM</div>
            <div class="insight-title">Plesetan "Makan Beracun Gratis"</div>
            <div class="insight-count">2.582 tweet</div>
            <div class="insight-desc">Publik mengubah akronim MBG dari "Makan Bergizi Gratis" menjadi "Makan Beracun Gratis". Reputasi rusak di level branding.</div>
        </div>
        <div class="insight-card">
            <div class="insight-tag">KOMUNIKASI KRISIS</div>
            <div class="insight-title">Persepsi "Berita Tiap Hari"</div>
            <div class="insight-count">1.857 tweet</div>
            <div class="insight-desc">Publik merasa berita keracunan terjadi setiap hari, menciptakan persepsi krisis berkelanjutan (news fatigue).</div>
        </div>
        <div class="insight-card">
            <div class="insight-tag">TRANSPARANSI DATA</div>
            <div class="insight-title">Fokus Angka Korban</div>
            <div class="insight-count">1.803 tweet</div>
            <div class="insight-desc">Diskursus berfokus pada jumlah korban ribuan jiwa. Publik mempertanyakan transparansi data resmi.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    colL, colR = st.columns([1, 1])

    with colL:
        st.markdown('<div class="section-title">6 Topik Hasil NMF</div>',
                    unsafe_allow_html=True)
        fig_topic = px.bar(
            df_topic_meta, x="size", y="suggested_name", orientation="h",
            text="size", color="suggested_name",
            color_discrete_sequence=px.colors.qualitative.Set2,
            hover_data={"top_terms": True, "share_pct": True,
                        "size": False, "suggested_name": False},
        )
        fig_topic.update_traces(textposition="outside", textfont_size=11)
        fig_topic.update_layout(
            yaxis={"categoryorder": "total ascending"},
            xaxis_title="", yaxis_title="",
            showlegend=False, height=340,
            margin=dict(t=20, b=20, l=10, r=40),
            plot_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_topic, use_container_width=True)

    with colR:
        st.markdown('<div class="section-title">Struktur Topik 2D</div>',
                    unsafe_allow_html=True)
        topic_id_to_name = dict(zip(df_topic_meta["topic_id"],
                                    df_topic_meta["suggested_name"]))
        df_topic_2d_display = df_topic_2d.copy()
        df_topic_2d_display["topic_name"] = df_topic_2d_display["topic_id"].map(topic_id_to_name)
        df_topic_2d_display["preview"] = df_topic_2d_display["full_text"].astype(str).str[:100] + "..."

        fig_2d = px.scatter(
            df_topic_2d_display, x="x", y="y", color="topic_name",
            color_discrete_sequence=px.colors.qualitative.Set2,
            hover_data={"preview": True, "x": False, "y": False, "topic_name": True},
            opacity=0.55,
        )
        fig_2d.update_traces(marker=dict(size=4))
        fig_2d.update_layout(
            xaxis_title="", yaxis_title="",
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            legend_title_text="", height=340,
            margin=dict(t=20, b=20, l=10, r=10),
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation="v", x=1.02, y=0.5, font=dict(size=10)),
        )
        st.plotly_chart(fig_2d, use_container_width=True)

    st.markdown('<div class="section-title">Konvergensi Aspek ↔ Topik</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">% overlap antara rule-based dan NMF. Sel pekat = dua '
        'metode sepakat.</div>',
        unsafe_allow_html=True,
    )

    overlap = compute_overlap(df_topic_assign, df_neg)
    if len(overlap) > 0:
        overlap = overlap.merge(df_topic_meta[["topic_id", "suggested_name"]],
                                on="topic_id", how="left")
        overlap["topik"] = overlap["suggested_name"]
        pivot = overlap.pivot(index="topik", columns="aspek", values="persen").fillna(0)
        pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

        fig_heat = px.imshow(
            pivot, text_auto=".1f", aspect="auto",
            color_continuous_scale=["#FFFFFF", "#DC2626"],
            labels=dict(x="", y="", color="%"),
        )
        fig_heat.update_layout(
            height=350,
            margin=dict(t=20, b=20, l=10, r=10),
            xaxis=dict(side="bottom", tickangle=-25),
        )
        st.plotly_chart(fig_heat, use_container_width=True)


# ---------------------------------------------------------------------
# TAB 3: REKOMENDASI
# ---------------------------------------------------------------------
with tab3:
    st.markdown(
        '<div class="section-sub">Rekomendasi per aspek, diurutkan dari yang paling banyak '
        'dikeluhkan. Klik untuk membuka.</div>',
        unsafe_allow_html=True,
    )

    sorted_aspects = aspect_counts.set_index("aspek").index.tolist()

    for asp_name in sorted_aspects:
        n = int(aspect_counts.loc[aspect_counts["aspek"] == asp_name, "jumlah"].iloc[0])
        pct = float(aspect_counts.loc[aspect_counts["aspek"] == asp_name, "persen"].iloc[0])
        with st.expander(f"**{asp_name}**  ·  {n:,} tweet  ·  {pct:.1f}%"):
            for i, rec in enumerate(ASPECTS[asp_name]["recommendations"], 1):
                st.markdown(f"**{i}.** {rec}")

    st.markdown('<div class="section-title">Rekomendasi Tambahan dari Topik AI</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Dimensi evaluasi yang ditemukan unsupervised NMF.</div>',
        unsafe_allow_html=True,
    )

    with st.expander("**Reputasi & Branding Program**  ·  2.582 tweet"):
        st.markdown("**1.** Strategi komunikasi positif dengan success story per daerah.")
        st.markdown("**2.** Crisis communication officer untuk respons cepat setiap insiden.")
        st.markdown("**3.** Engage influencer pendidikan dan ahli gizi sebagai third-party endorsement.")
        st.markdown("**4.** Evaluasi rebranding atau tagline yang menetralkan plesetan negatif.")

    with st.expander("**Komunikasi Krisis & News Fatigue**  ·  1.857 tweet"):
        st.markdown("**1.** Laporan agregat bulanan: total dapur, total porsi, total insiden + rasio.")
        st.markdown("**2.** Counter-narrative berbasis data ('X juta porsi disajikan, Y% rate insiden').")
        st.markdown("**3.** Konten substantif di media sosial, bukan defensif.")
        st.markdown("**4.** Press briefing rutin untuk wartawan dan kreator konten pendidikan.")

    with st.expander("**Transparansi Data Korban**  ·  1.803 tweet"):
        st.markdown("**1.** Dashboard publik real-time: jumlah insiden, korban per insiden, status investigasi.")
        st.markdown("**2.** Definisi 'korban' konsisten dan diumumkan publik.")
        st.markdown("**3.** Update mingguan, jangan menunggu media menghitung sendiri.")
        st.markdown("**4.** Kolaborasi Kemenkes untuk validasi angka resmi.")


# ---------------------------------------------------------------------
# TAB 4: EKSPLORASI
# ---------------------------------------------------------------------
with tab4:
    sub1, sub2 = st.tabs(["Per Aspek", "Per Topik AI"])

    with sub1:
        c1, c2 = st.columns([3, 1])
        with c1:
            selected_aspect = st.selectbox(
                "Pilih aspek", list(ASPECTS.keys()),
                key="exp_aspect", label_visibility="collapsed",
            )
        with c2:
            n_samples = st.number_input(
                "Sample", 5, 30, 10, 5,
                key="n_asp", label_visibility="collapsed",
            )

        df_a = df_neg[df_neg["aspects"].apply(lambda lst: selected_aspect in lst)].copy()
        df_a = df_a.sort_values("confidence_indobert", ascending=False)
        st.caption(f"{len(df_a):,} tweet · menampilkan {min(n_samples, len(df_a))} teratas")

        for _, row in df_a.head(n_samples).iterrows():
            others = [a for a in row["aspects"] if a != selected_aspect]
            meta = f"Confidence {row['confidence_indobert']:.2f}"
            if others:
                meta += f"  ·  juga: {', '.join(others)}"
            text = str(row['full_text'])[:400].replace("<", "&lt;").replace(">", "&gt;")
            st.markdown(
                f'<div class="tweet-box">{text}<div class="tweet-meta">{meta}</div></div>',
                unsafe_allow_html=True,
            )

    with sub2:
        c1, c2 = st.columns([3, 1])
        with c1:
            topic_options = dict(zip(
                df_topic_meta["suggested_name"], df_topic_meta["topic_id"]
            ))
            sel_topic_label = st.selectbox(
                "Pilih topik", list(topic_options.keys()),
                key="exp_topic", label_visibility="collapsed",
            )
            sel_topic_id = topic_options[sel_topic_label]
        with c2:
            n_samples_t = st.number_input(
                "Sample", 5, 30, 10, 5,
                key="n_topic", label_visibility="collapsed",
            )

        df_t = df_topic_assign[df_topic_assign["dominant_topic"] == sel_topic_id].copy()
        df_t = df_t.sort_values("topic_score", ascending=False)
        st.caption(f"{len(df_t):,} tweet · menampilkan {min(n_samples_t, len(df_t))} teratas")

        for _, row in df_t.head(n_samples_t).iterrows():
            meta = (f"Topic score {row['topic_score']:.3f}  ·  "
                    f"Confidence {row['confidence_indobert']:.2f}")
            text = str(row['full_text'])[:400].replace("<", "&lt;").replace(">", "&gt;")
            st.markdown(
                f'<div class="tweet-box">{text}<div class="tweet-meta">{meta}</div></div>',
                unsafe_allow_html=True,
            )


# =====================================================================
# FOOTER
# =====================================================================
st.markdown("---")
with st.expander("Metodologi & batasan"):
    st.markdown("""
**Pipeline.** IndoBERT + TF-IDF/SVM untuk klasifikasi sentimen → rule-based keyword mapping
(7 aspek operasional, multi-label) → NMF unsupervised topic modeling (6 topik dari TF-IDF
tweet negatif, max_features=2500, max_df=0.35, sublinear_tf, L2-normalized) →
cross-validation aspek ↔ topik via document join.

**Batasan.** Dataset tidak memuat kolom tanggal (tidak ada time series). Tweet pendek
menyebabkan satu cluster dominan (Kasus Umum, 63.8%); 5 topik minoritas tetap memberi
insight terdiferensiasi. Rule-based keyword sensitif terhadap sarkasme dan bahasa
daerah. Random seed: 42 (reproducible).
""")
