import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier


# ============================================================
# 1. PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="MindTrack UB Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. GLOBAL STYLE
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }

    .stApp {
        background: radial-gradient(circle at top left, #18213f 0%, #0B0F19 35%, #070A12 100%);
        color: #F8FAFC;
    }

    .block-container {
        padding-top: 2.6rem !important;
        padding-bottom: 3rem !important;
        max-width: 1450px;
    }

    [data-testid="stHeader"] {
        background: rgba(7, 10, 18, 0.84);
        backdrop-filter: blur(10px);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #151926 0%, #101420 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    [data-testid="stSidebar"] * {
        color: #F8FAFC !important;
    }

    .hero {
        margin-top: 0.6rem;
        padding: 2.3rem 2.4rem;
        border-radius: 26px;
        background:
            linear-gradient(135deg, rgba(124, 58, 237, 0.92), rgba(6, 182, 212, 0.82)),
            radial-gradient(circle at top right, rgba(255,255,255,0.30), transparent 40%);
        box-shadow: 0 25px 70px rgba(0,0,0,0.35);
        border: 1px solid rgba(255,255,255,0.16);
        overflow: hidden;
    }

    .hero h1 {
        color: #FFFFFF !important;
        font-size: 2.65rem;
        font-weight: 850;
        margin: 0 0 0.8rem 0;
        letter-spacing: -0.04em;
    }

    .hero p {
        color: #ECFEFF !important;
        font-size: 1.04rem;
        line-height: 1.7;
        max-width: 940px;
        margin: 0;
    }

    .section-title {
        margin-top: 1.4rem;
        margin-bottom: 1rem;
        font-size: 1.75rem;
        font-weight: 800;
        color: #F8FAFC !important;
        letter-spacing: -0.03em;
    }

    .section-subtitle {
        color: #CBD5E1 !important;
        font-size: 0.98rem;
        line-height: 1.65;
        margin-bottom: 1rem;
    }

    .metric-card {
        padding: 1.25rem 1.3rem;
        border-radius: 22px;
        background: linear-gradient(180deg, rgba(30,41,59,0.88), rgba(15,23,42,0.88));
        border: 1px solid rgba(148,163,184,0.18);
        box-shadow: 0 18px 38px rgba(0,0,0,0.28);
        min-height: 150px;
    }

    .metric-card .label {
        color: #CBD5E1 !important;
        font-size: 0.86rem;
        font-weight: 650;
        margin-bottom: 0.65rem;
    }

    .metric-card .value {
        color: #FFFFFF !important;
        font-size: 2.05rem;
        font-weight: 850;
        line-height: 1.08;
        margin-bottom: 0.5rem;
        word-break: break-word;
    }

    .metric-card .hint {
        color: #94A3B8 !important;
        font-size: 0.82rem;
        line-height: 1.45;
    }

    .metric-card .accent {
        width: 42px;
        height: 5px;
        border-radius: 999px;
        margin-top: 0.95rem;
    }

    .callout {
        padding: 1.05rem 1.25rem;
        border-radius: 20px;
        background: linear-gradient(90deg, rgba(79,70,229,0.24), rgba(6,182,212,0.20));
        border-left: 5px solid #8B5CF6;
        border-top: 1px solid rgba(255,255,255,0.10);
        border-bottom: 1px solid rgba(255,255,255,0.08);
        margin: 1rem 0 1.4rem 0;
        color: #F8FAFC !important;
        box-shadow: 0 14px 35px rgba(0,0,0,0.22);
    }

    .callout strong {
        color: #FFFFFF !important;
        font-weight: 800;
    }

    .callout code {
        color: #FDE68A !important;
        background: rgba(15,23,42,0.55);
        padding: 0.12rem 0.35rem;
        border-radius: 8px;
    }

    .light-callout {
        padding: 1.05rem 1.25rem;
        border-radius: 20px;
        background: linear-gradient(90deg, #F8FAFC, #E0F2FE);
        border-left: 5px solid #06B6D4;
        margin: 1rem 0 1.4rem 0;
        color: #0F172A !important;
        box-shadow: 0 14px 35px rgba(0,0,0,0.20);
    }

    .light-callout strong,
    .light-callout span,
    .light-callout p {
        color: #0F172A !important;
    }

    .mini-card {
        padding: 1.15rem 1.25rem;
        border-radius: 20px;
        background: rgba(15, 23, 42, 0.76);
        border: 1px solid rgba(148,163,184,0.16);
        min-height: 150px;
    }

    .mini-card h4 {
        color: #FFFFFF !important;
        font-weight: 800;
        margin-bottom: 0.45rem;
    }

    .mini-card p {
        color: #CBD5E1 !important;
        line-height: 1.6;
        font-size: 0.92rem;
    }

    .risk-low {
        background: rgba(34,197,94,0.13);
        border: 1px solid rgba(34,197,94,0.35);
        color: #DCFCE7 !important;
    }

    .risk-medium {
        background: rgba(245,158,11,0.14);
        border: 1px solid rgba(245,158,11,0.35);
        color: #FEF3C7 !important;
    }

    .risk-high {
        background: rgba(239,68,68,0.14);
        border: 1px solid rgba(239,68,68,0.35);
        color: #FEE2E2 !important;
    }

    .risk-box {
        padding: 1.1rem 1.2rem;
        border-radius: 18px;
        margin-top: 0.8rem;
    }

    .risk-box h3 {
        margin: 0 0 0.5rem 0;
        font-weight: 850;
        color: inherit !important;
    }

    .risk-box p, .risk-box li {
        color: inherit !important;
        line-height: 1.55;
    }

    div[data-testid="stMetric"] {
        background: rgba(15,23,42,0.72);
        padding: 1rem;
        border-radius: 18px;
        border: 1px solid rgba(148,163,184,0.18);
    }

    div[data-testid="stTabs"] button p {
        color: #E5E7EB !important;
        font-weight: 700;
    }

    div[data-testid="stTabs"] button[aria-selected="true"] p {
        color: #FFFFFF !important;
    }

    .stPlotlyChart {
        background: rgba(15, 23, 42, 0.45);
        border-radius: 20px;
        border: 1px solid rgba(148,163,184,0.10);
        padding: 0.8rem;
    }

    label, .stMarkdown, p, li {
        color: #F8FAFC;
    }

    .stButton > button {
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.18);
        background: linear-gradient(135deg, #7C3AED, #06B6D4);
        color: #FFFFFF;
        font-weight: 800;
        padding: 0.65rem 1rem;
    }

    .stCheckbox label p,
    .stRadio label p {
        color: #E5E7EB !important;
        font-weight: 600;
    }

    div[data-baseweb="select"] > div {
        background-color: #0F172A !important;
        border-color: rgba(148,163,184,0.35) !important;
    }

    div[data-baseweb="select"] span {
        color: #F8FAFC !important;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 18px;
        overflow: hidden;
        border: 1px solid rgba(148,163,184,0.18);
    }

    hr {
        border-color: rgba(148,163,184,0.18);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. DATA LOADING & CLEANING
# ============================================================

DATA_CANDIDATES = [
    Path(__file__).resolve().parent.parent / "data" / "Student Mental health.csv",
    Path(__file__).resolve().parent / "data" / "Student Mental health.csv",
    Path.cwd() / "data" / "Student Mental health.csv",
    Path.cwd() / "Student Mental health.csv",
]


def find_dataset_path():
    for path in DATA_CANDIDATES:
        if path.exists():
            return path
    return None


@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        raw = pd.read_csv(uploaded_file)
    else:
        dataset_path = find_dataset_path()
        if dataset_path is None:
            return None
        raw = pd.read_csv(dataset_path)

    df = raw.copy()

    rename_map = {
        "Timestamp": "timestamp",
        "Choose your gender": "gender",
        "Age": "age",
        "What is your course?": "course",
        "Your current year of Study": "year_of_study",
        "What is your CGPA?": "cgpa",
        "Marital status": "marital_status",
        "Do you have Depression?": "depression",
        "Do you have Anxiety?": "anxiety",
        "Do you have Panic attack?": "panic_attack",
        "Did you seek any specialist for a treatment?": "seek_treatment",
    }

    df = df.rename(columns=rename_map)

    needed_cols = [
        "gender",
        "age",
        "course",
        "year_of_study",
        "cgpa",
        "marital_status",
        "depression",
        "anxiety",
        "panic_attack",
        "seek_treatment",
    ]

    for col in needed_cols:
        if col not in df.columns:
            df[col] = np.nan

    df["gender"] = df["gender"].fillna("Unknown").astype(str).str.strip().str.title()
    df["course"] = (
        df["course"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.title()
    )

    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    if df["age"].notna().sum() > 0:
        df["age"] = df["age"].fillna(df["age"].median())
    else:
        df["age"] = df["age"].fillna(0)

    def clean_year(value):
        text = str(value).strip().title()
        extracted = pd.Series([text]).str.extract(r"(\d+)").iloc[0, 0]
        if pd.notna(extracted):
            return f"Year {int(extracted)}"
        return text if text and text != "Nan" else "Unknown"

    df["year_of_study"] = df["year_of_study"].apply(clean_year)

    df["cgpa"] = (
        df["cgpa"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    df["marital_status"] = (
        df["marital_status"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .str.title()
    )

    def clean_yes_no(value):
        text = str(value).strip().title()
        if text in ["Yes", "Y", "True", "1"]:
            return "Yes"
        if text in ["No", "N", "False", "0"]:
            return "No"
        return "No"

    for col in ["depression", "anxiety", "panic_attack", "seek_treatment"]:
        df[col] = df[col].apply(clean_yes_no)

    df["depression_flag"] = (df["depression"] == "Yes").astype(int)
    df["anxiety_flag"] = (df["anxiety"] == "Yes").astype(int)
    df["panic_flag"] = (df["panic_attack"] == "Yes").astype(int)

    df["symptom_count"] = df[["depression_flag", "anxiety_flag", "panic_flag"]].sum(axis=1)

    def risk_label(score):
        if score == 0:
            return "Low Risk"
        if score == 1:
            return "Medium Risk"
        return "High Risk"

    df["risk_level"] = df["symptom_count"].apply(risk_label)
    df["risk_score"] = df["symptom_count"] / 3
    df["has_any_indicator"] = np.where(df["symptom_count"] > 0, "Has Indicator", "No Indicator")
    df["binary_risk"] = np.where(df["symptom_count"] > 0, "At Risk", "No Risk")
    df["treatment_gap"] = np.where(
        (df["symptom_count"] > 0) & (df["seek_treatment"] == "No"),
        "At Risk, No Treatment",
        "Other",
    )

    return df


df = load_data()

if df is None:
    st.error("Dataset tidak ditemukan. Upload file `Student Mental health.csv` dulu untuk menjalankan dashboard.")
    uploaded_file = st.file_uploader("Upload dataset CSV", type=["csv"])
    if uploaded_file is not None:
        df = load_data(uploaded_file)
    else:
        st.stop()


# ============================================================
# 4. MODEL TRAINING FOR SIMULATION
# ============================================================

@st.cache_resource
def train_screening_model(data):
    model_df = data.copy()

    feature_cols = [
        "gender",
        "age",
        "course",
        "year_of_study",
        "cgpa",
        "marital_status",
    ]

    X = model_df[feature_cols].copy()
    y = np.where(model_df["symptom_count"] > 0, 1, 0)

    numeric_features = ["age"]
    categorical_features = [
        "gender",
        "course",
        "year_of_study",
        "cgpa",
        "marital_status",
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = GradientBoostingClassifier(random_state=42)

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X, y)

    return pipeline, feature_cols


screening_model, screening_feature_cols = train_screening_model(df)


# ============================================================
# 5. CONSTANTS & HELPER FUNCTIONS
# ============================================================

RISK_ORDER = ["Low Risk", "Medium Risk", "High Risk"]
RISK_COLORS = {
    "Low Risk": "#22C55E",
    "Medium Risk": "#F59E0B",
    "High Risk": "#EF4444",
}
YES_NO_COLORS = {
    "Yes": "#06B6D4",
    "No": "#EF4444",
}
BINARY_COLORS = {
    "No Risk": "#22C55E",
    "At Risk": "#EF4444",
}
INDICATOR_COLORS = {
    "Depression": "#A78BFA",
    "Anxiety": "#22D3EE",
    "Panic Attack": "#FB7185",
}
CGPA_ORDER = ["0 - 1.99", "2.00 - 2.49", "2.50 - 2.99", "3.00 - 3.49", "3.50 - 4.00", "Unknown"]


def safe_pct(part, total):
    if total == 0:
        return 0
    return round((part / total) * 100, 1)


def sort_years(values):
    def key_func(x):
        digits = "".join([c for c in str(x) if c.isdigit()])
        return int(digits) if digits else 99

    return sorted(values, key=key_func)


def style_fig(fig, height=390):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F8FAFC", family="Inter, sans-serif", size=13),
        margin=dict(l=20, r=20, t=60, b=35),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#F8FAFC"),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.12)",
        zerolinecolor="rgba(148,163,184,0.18)",
        tickfont=dict(color="#CBD5E1"),
        title_font=dict(color="#F8FAFC"),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.12)",
        zerolinecolor="rgba(148,163,184,0.18)",
        tickfont=dict(color="#CBD5E1"),
        title_font=dict(color="#F8FAFC"),
    )
    return fig


def metric_card(label, value, hint, color="#8B5CF6"):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="hint">{hint}</div>
            <div class="accent" style="background:{color};"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(title, subtitle=None):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def callout(text, light=False):
    class_name = "light-callout" if light else "callout"
    st.markdown(f'<div class="{class_name}">{text}</div>', unsafe_allow_html=True)


def pill_filter(label, options, key, default=None):
    options = [str(x) for x in options if str(x) != "nan"]
    if default is None:
        default = options

    st.sidebar.markdown(f"**{label}**")

    if hasattr(st, "pills"):
        selected = st.sidebar.pills(
            label=f"{label} filter",
            options=options,
            default=default,
            selection_mode="multi",
            key=key,
            label_visibility="collapsed",
        )
        return selected if selected else []

    selected = []
    cols = st.sidebar.columns(2)
    for idx, option in enumerate(options):
        with cols[idx % 2]:
            checked = st.checkbox(option, value=option in default, key=f"{key}_{option}")
            if checked:
                selected.append(option)
    return selected


def filter_dataframe(data):
    st.sidebar.markdown("## 🔎 Filter Dashboard")
    st.sidebar.caption("Filter ini memengaruhi grafik utama, bukan simulasi model.")

    gender_options = sorted(data["gender"].dropna().unique())
    year_options = sort_years(data["year_of_study"].dropna().unique())
    cgpa_options = [x for x in CGPA_ORDER if x in data["cgpa"].unique()]
    risk_options = RISK_ORDER

    selected_gender = pill_filter("Gender", gender_options, "filter_gender")
    selected_year = pill_filter("Year of Study", year_options, "filter_year")
    selected_cgpa = pill_filter("CGPA Range", cgpa_options, "filter_cgpa")
    selected_risk = pill_filter("Risk Level", risk_options, "filter_risk")

    filtered = data[
        data["gender"].isin(selected_gender)
        & data["year_of_study"].isin(selected_year)
        & data["cgpa"].isin(selected_cgpa)
        & data["risk_level"].isin(selected_risk)
    ].copy()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📌 Catatan")
    st.sidebar.info(
        "Dashboard ini adalah prototype berbasis dataset kecil. Output dipakai sebagai sinyal awal, bukan diagnosis medis."
    )

    return filtered


filtered_df = filter_dataframe(df)


# ============================================================
# 6. HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>MindTrack UB</h1>
        <p>
            Prototype dashboard early warning untuk memetakan risiko kesehatan mental mahasiswa,
            membaca treatment gap, menampilkan evidence modeling, dan menjalankan simulasi prediksi
            berbasis model machine learning.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if filtered_df.empty:
    st.warning("Tidak ada data yang sesuai dengan filter saat ini. Pilih kembali minimal satu opsi pada setiap filter.")
    st.stop()


# ============================================================
# 7. SUMMARY METRICS
# ============================================================

total_students = len(filtered_df)
depression_count = int((filtered_df["depression"] == "Yes").sum())
anxiety_count = int((filtered_df["anxiety"] == "Yes").sum())
panic_count = int((filtered_df["panic_attack"] == "Yes").sum())
seek_count = int((filtered_df["seek_treatment"] == "Yes").sum())
at_risk_count = int((filtered_df["symptom_count"] > 0).sum())
gap_count = int(((filtered_df["symptom_count"] > 0) & (filtered_df["seek_treatment"] == "No")).sum())

st.markdown("<br>", unsafe_allow_html=True)

m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    metric_card("Total Responden", total_students, "Jumlah data setelah filter aktif.", "#8B5CF6")
with m2:
    metric_card("Memiliki Indikator", at_risk_count, f"{safe_pct(at_risk_count, total_students)}% dari data terfilter.", "#06B6D4")
with m3:
    metric_card("High Risk", int((filtered_df["risk_level"] == "High Risk").sum()), "Memiliki 2–3 indikator mental health.", "#EF4444")
with m4:
    metric_card("Seek Treatment", seek_count, f"{safe_pct(seek_count, total_students)}% mencari bantuan spesialis.", "#22C55E")
with m5:
    metric_card("Treatment Gap", gap_count, "Berisiko tetapi belum mencari bantuan.", "#F59E0B")


# ============================================================
# 8. TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "📌 Executive Overview",
        "🎯 Risk Segmentation",
        "🧩 Treatment Gap",
        "📊 Pattern Analysis",
        "🤖 Model Evidence",
        "🛟 ML Screening Simulation",
        "🔐 Ethics & Limits",
    ]
)


# ============================================================
# TAB 1 - EXECUTIVE OVERVIEW
# ============================================================

with tab1:
    section_title("📌 Executive Overview")
    callout(
        """
        <strong>Core insight:</strong>
        sistem ini memosisikan data sebagai sinyal awal, bukan sebagai diagnosis.
        Tujuannya membantu kampus melihat kelompok yang membutuhkan perhatian,
        lalu mengarahkan tindak lanjut ke layanan yang paling sesuai.
        """
    )

    col1, col2 = st.columns([1.1, 1])

    with col1:
        indicator_df = pd.DataFrame(
            {
                "Indicator": ["Depression", "Anxiety", "Panic Attack"],
                "Count": [depression_count, anxiety_count, panic_count],
            }
        )

        fig = px.bar(
            indicator_df,
            x="Indicator",
            y="Count",
            text="Count",
            color="Indicator",
            color_discrete_map=INDICATOR_COLORS,
            title="Jumlah Indikator Mental Health yang Terdeteksi",
        )
        fig.update_traces(textposition="outside", marker_line_width=0)
        fig = style_fig(fig, height=390)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        risk_count_df = (
            filtered_df["risk_level"]
            .value_counts()
            .reindex(RISK_ORDER, fill_value=0)
            .reset_index()
        )
        risk_count_df.columns = ["Risk Level", "Count"]

        fig = px.pie(
            risk_count_df,
            names="Risk Level",
            values="Count",
            hole=0.58,
            color="Risk Level",
            color_discrete_map=RISK_COLORS,
            title="Komposisi Risk Level",
        )
        fig.update_traces(
            textinfo="percent+label",
            textfont=dict(color="#FFFFFF", size=13),
            marker=dict(line=dict(color="rgba(255,255,255,0.18)", width=1)),
        )
        fig = style_fig(fig, height=390)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Insight Ringkas")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="mini-card">
                <h4>1. Risiko tidak selalu terlihat langsung</h4>
                <p>
                    Indikator seperti depression, anxiety, dan panic attack dapat dipakai
                    sebagai sinyal awal untuk memahami kebutuhan pendampingan mahasiswa.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="mini-card">
                <h4>2. Treatment gap perlu diperhatikan</h4>
                <p>
                    Ada mahasiswa yang sudah menunjukkan indikator risiko, tetapi belum mencari bantuan.
                    Ini memperkuat kebutuhan pendekatan yang lebih proaktif.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            """
            <div class="mini-card">
                <h4>3. Dashboard membantu prioritisasi</h4>
                <p>
                    Kampus dapat membaca pola secara agregat, bukan menilai mahasiswa secara sembarangan.
                    Keputusan akhir tetap berada pada konselor atau pihak berwenang.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# TAB 2 - RISK SEGMENTATION
# ============================================================

with tab2:
    section_title(
        "🎯 Risk Segmentation",
        "Segmentasi risiko dibuat dari hasil analisis indikator mental health pada dataset.",
    )

    callout(
        """
        <strong>Definisi analitik:</strong>
        Low Risk = 0 indikator, Medium Risk = 1 indikator, High Risk = 2–3 indikator.
        Segmentasi ini digunakan untuk membaca pola data dan prioritas pendampingan,
        sedangkan simulasi prediksi menggunakan model machine learning pada tab ML Screening Simulation.
        """
    )

    risk_summary = (
        filtered_df.groupby("risk_level")
        .agg(
            total=("risk_level", "count"),
            avg_symptom=("symptom_count", "mean"),
            seek_treatment=("seek_treatment", lambda x: (x == "Yes").sum()),
        )
        .reindex(RISK_ORDER)
        .reset_index()
        .fillna(0)
    )

    risk_summary["avg_symptom"] = risk_summary["avg_symptom"].round(2)
    risk_summary["treatment_rate"] = risk_summary.apply(
        lambda row: safe_pct(row["seek_treatment"], row["total"]), axis=1
    )

    col1, col2 = st.columns([1.15, 1])

    with col1:
        fig = px.bar(
            risk_summary,
            x="risk_level",
            y="total",
            text="total",
            color="risk_level",
            color_discrete_map=RISK_COLORS,
            title="Jumlah Mahasiswa per Risk Level",
            labels={"risk_level": "Risk Level", "total": "Jumlah"},
        )
        fig.update_traces(textposition="outside")
        fig = style_fig(fig, height=410)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            risk_summary,
            x="risk_level",
            y="treatment_rate",
            text="treatment_rate",
            color="risk_level",
            color_discrete_map=RISK_COLORS,
            title="Treatment Rate per Risk Level (%)",
            labels={"risk_level": "Risk Level", "treatment_rate": "Treatment Rate (%)"},
        )
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig = style_fig(fig, height=410)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Tabel Ringkasan Segmentasi")
    st.dataframe(
        risk_summary.rename(
            columns={
                "risk_level": "Risk Level",
                "total": "Total",
                "avg_symptom": "Avg Symptom Count",
                "seek_treatment": "Seek Treatment",
                "treatment_rate": "Treatment Rate (%)",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# TAB 3 - TREATMENT GAP
# ============================================================

with tab3:
    section_title("🧩 Treatment Gap Analysis")
    callout(
        """
        <strong>Treatment gap</strong> adalah kondisi ketika mahasiswa sudah menunjukkan indikator risiko,
        tetapi belum mencari bantuan spesialis. Bagian ini menjadi alasan kenapa sistem yang hanya menunggu
        mahasiswa datang sendiri berpotensi kurang optimal.
        """
    )

    treatment_by_risk = (
        filtered_df.groupby(["risk_level", "seek_treatment"])
        .size()
        .reset_index(name="count")
    )

    fig = px.bar(
        treatment_by_risk,
        x="risk_level",
        y="count",
        color="seek_treatment",
        barmode="group",
        category_orders={"risk_level": RISK_ORDER, "seek_treatment": ["No", "Yes"]},
        color_discrete_map=YES_NO_COLORS,
        title="Treatment Status per Risk Level",
        labels={
            "risk_level": "Risk Level",
            "count": "Jumlah Mahasiswa",
            "seek_treatment": "Seek Treatment",
        },
        text="count",
    )
    fig.update_traces(textposition="outside")
    fig = style_fig(fig, height=430)
    st.plotly_chart(fig, use_container_width=True)

    gap_df = filtered_df[filtered_df["treatment_gap"] == "At Risk, No Treatment"]

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card(
            "Mahasiswa Berisiko",
            at_risk_count,
            "Memiliki minimal satu indikator mental health.",
            "#06B6D4",
        )
    with c2:
        metric_card(
            "Treatment Gap",
            len(gap_df),
            f"{safe_pct(len(gap_df), at_risk_count)}% dari mahasiswa berisiko belum mencari bantuan.",
            "#F59E0B",
        )
    with c3:
        metric_card(
            "Sudah Seek Treatment",
            int(((filtered_df["symptom_count"] > 0) & (filtered_df["seek_treatment"] == "Yes")).sum()),
            "Mahasiswa berisiko yang sudah mencari bantuan spesialis.",
            "#22C55E",
        )

    st.markdown("### Interpretasi")
    st.markdown(
        """
        Treatment gap menunjukkan bahwa keberadaan layanan saja belum otomatis membuat mahasiswa mencari bantuan.
        Karena itu, sistem pendukung keputusan dapat membantu kampus melakukan pendekatan yang lebih proaktif,
        misalnya melalui edukasi, self-assessment lanjutan, peer counselor, atau prioritas outreach oleh konselor.
        """
    )


# ============================================================
# TAB 4 - PATTERN ANALYSIS
# ============================================================

with tab4:
    section_title(
        "📊 Pattern Analysis",
        "Bagian ini melihat pola risiko berdasarkan atribut demografis dan akademik yang tersedia di dataset.",
    )

    col1, col2 = st.columns(2)

    with col1:
        gender_risk = (
            filtered_df.groupby(["gender", "risk_level"])
            .size()
            .reset_index(name="count")
        )
        fig = px.bar(
            gender_risk,
            x="gender",
            y="count",
            color="risk_level",
            barmode="group",
            category_orders={"risk_level": RISK_ORDER},
            color_discrete_map=RISK_COLORS,
            title="Risk Level berdasarkan Gender",
            labels={"gender": "Gender", "count": "Jumlah", "risk_level": "Risk Level"},
            text="count",
        )
        fig.update_traces(textposition="outside")
        fig = style_fig(fig, height=410)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        year_risk = (
            filtered_df.groupby(["year_of_study", "risk_level"])
            .size()
            .reset_index(name="count")
        )
        fig = px.bar(
            year_risk,
            x="year_of_study",
            y="count",
            color="risk_level",
            barmode="group",
            category_orders={
                "risk_level": RISK_ORDER,
                "year_of_study": sort_years(filtered_df["year_of_study"].unique()),
            },
            color_discrete_map=RISK_COLORS,
            title="Risk Level berdasarkan Tahun Studi",
            labels={"year_of_study": "Year of Study", "count": "Jumlah", "risk_level": "Risk Level"},
            text="count",
        )
        fig.update_traces(textposition="outside")
        fig = style_fig(fig, height=410)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        cgpa_risk = (
            filtered_df.groupby(["cgpa", "risk_level"])
            .size()
            .reset_index(name="count")
        )
        fig = px.bar(
            cgpa_risk,
            x="cgpa",
            y="count",
            color="risk_level",
            barmode="stack",
            category_orders={"risk_level": RISK_ORDER, "cgpa": CGPA_ORDER},
            color_discrete_map=RISK_COLORS,
            title="Risk Level berdasarkan CGPA",
            labels={"cgpa": "CGPA Range", "count": "Jumlah", "risk_level": "Risk Level"},
        )
        fig = style_fig(fig, height=410)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        course_risk = (
            filtered_df[filtered_df["risk_level"].isin(["Medium Risk", "High Risk"])]
            .groupby("course")
            .size()
            .sort_values(ascending=False)
            .head(10)
            .reset_index(name="count")
        )

        if course_risk.empty:
            st.info("Tidak ada data Medium/High Risk pada filter saat ini.")
        else:
            fig = px.bar(
                course_risk,
                x="count",
                y="course",
                orientation="h",
                title="Top 10 Course dengan Medium/High Risk",
                labels={"count": "Jumlah", "course": "Course"},
                text="count",
                color="count",
                color_continuous_scale="Turbo",
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(coloraxis_showscale=False)
            fig = style_fig(fig, height=410)
            st.plotly_chart(fig, use_container_width=True)

    callout(
        """
        <strong>Catatan interpretasi:</strong>
        pola pada grafik tidak boleh dipakai untuk menstigma gender, jurusan, atau angkatan tertentu.
        Analisis ini hanya membantu membaca area yang mungkin membutuhkan dukungan lebih lanjut.
        """
    )


# ============================================================
# TAB 5 - MODEL EVIDENCE
# ============================================================

with tab5:
    section_title(
        "🤖 Model Evidence",
        "Bagian ini menjelaskan hasil eksperimen modeling terbaru dan bagaimana model diposisikan dalam sistem.",
    )

    callout(
        """
        <strong>Kesimpulan modeling:</strong>
        pendekatan terbaik yang valid adalah Binary Classification dengan target
        <strong>No Risk vs At Risk</strong>. Model ini lebih stabil dibandingkan klasifikasi
        tiga kelas Low / Medium / High karena dataset kecil dan fitur yang tersedia masih terbatas.
        """
    )

    st.markdown("### Best Valid Model")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "Best Model",
            "Gradient Boosting",
            "Model terbaik pada eksperimen valid.",
            "#8B5CF6",
        )

    with c2:
        metric_card(
            "Accuracy CV",
            "0.702",
            "Rata-rata accuracy dari cross-validation.",
            "#06B6D4",
        )

    with c3:
        metric_card(
            "Balanced Acc.",
            "0.633",
            "Lebih adil untuk data yang tidak seimbang.",
            "#F59E0B",
        )

    with c4:
        metric_card(
            "F1 Macro",
            "0.630",
            "Metrik utama untuk membaca performa antar kelas.",
            "#22C55E",
        )

    st.markdown("### Perbandingan Eksperimen Utama")

    model_result_df = pd.DataFrame(
        {
            "Experiment": [
                "Binary At Risk",
                "Binary At Risk with Oversampling",
                "Binary At Risk with Oversampling",
                "Binary At Risk",
                "Binary At Risk",
                "Binary At Risk",
                "Binary At Risk",
                "Multiclass Risk Level",
                "Multiclass Risk Level",
                "Multiclass Risk Level",
                "Multiclass Risk Level",
                "Dummy Baseline",
            ],
            "Model": [
                "Gradient Boosting",
                "LogReg Balanced + ROS",
                "RF + ROS",
                "Logistic Regression",
                "Logistic Regression Balanced",
                "Decision Tree Balanced",
                "Extra Trees Balanced",
                "Extra Trees Balanced",
                "Gradient Boosting",
                "Logistic Regression Balanced",
                "Random Forest Balanced",
                "Dummy Most Frequent",
            ],
            "Accuracy Mean": [
                0.702,
                0.642,
                0.613,
                0.683,
                0.593,
                0.573,
                0.564,
                0.515,
                0.445,
                0.436,
                0.406,
                0.634,
            ],
            "Balanced Accuracy Mean": [
                0.633,
                0.613,
                0.607,
                0.600,
                0.585,
                0.610,
                0.607,
                0.517,
                0.448,
                0.433,
                0.404,
                0.500,
            ],
            "F1 Macro Mean": [
                0.630,
                0.612,
                0.596,
                0.576,
                0.574,
                0.570,
                0.548,
                0.516,
                0.442,
                0.423,
                0.386,
                0.388,
            ],
            "F1 Weighted Mean": [
                0.674,
                0.639,
                0.615,
                0.634,
                0.595,
                0.572,
                0.549,
                0.507,
                0.440,
                0.426,
                0.385,
                0.492,
            ],
        }
    )

    fig = px.bar(
        model_result_df.sort_values("F1 Macro Mean", ascending=True),
        x="F1 Macro Mean",
        y="Model",
        color="Experiment",
        orientation="h",
        text="F1 Macro Mean",
        title="Perbandingan Model berdasarkan F1 Macro",
        labels={
            "F1 Macro Mean": "F1 Macro Mean",
            "Model": "Model",
            "Experiment": "Experiment",
        },
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig = style_fig(fig, height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        model_result_df.sort_values("F1 Macro Mean", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Kenapa Binary Model Lebih Dipilih?")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="mini-card">
                <h4>Multiclass terlalu berat untuk dataset kecil</h4>
                <p>
                    Target Low, Medium, dan High Risk membuat model harus membedakan tiga kelas
                    dengan jumlah data yang terbatas. Akibatnya, performa model cenderung tidak stabil,
                    terutama pada kelas Medium Risk.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="mini-card">
                <h4>Binary lebih sesuai untuk screening awal</h4>
                <p>
                    Untuk kebutuhan early warning, pertanyaan awal yang paling penting adalah apakah
                    mahasiswa masuk kelompok berisiko atau tidak. Detail Low, Medium, dan High tetap
                    digunakan sebagai hasil analisis prioritas pada dashboard.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Feature Importance: Valid vs Leakage")

    feature_importance_valid = pd.DataFrame(
        {
            "Feature": [
                "course / course category",
                "marital status",
                "year of study",
                "CGPA",
                "age",
                "gender",
            ],
            "Interpretation": [
                "Memberi sinyal tidak langsung terkait konteks akademik.",
                "Dapat berkaitan dengan tekanan personal, tetapi tidak boleh disimpulkan berlebihan.",
                "Merepresentasikan fase studi yang berbeda.",
                "Merepresentasikan performa akademik secara umum.",
                "Merepresentasikan rentang usia mahasiswa.",
                "Dipakai sebagai atribut demografis, bukan dasar stereotip.",
            ],
        }
    )

    st.dataframe(
        feature_importance_valid,
        use_container_width=True,
        hide_index=True,
    )

    leakage_df = pd.DataFrame(
        {
            "Scenario": [
                "Valid Modeling",
                "Leakage Modeling",
            ],
            "Feature Used": [
                "Gender, age, course, year of study, CGPA, marital status",
                "Depression, anxiety, panic attack ikut dimasukkan sebagai fitur",
            ],
            "Interpretation": [
                "Lebih valid, tetapi performanya terbatas karena fitur hanya sinyal tidak langsung.",
                "Skor bisa sangat tinggi, tetapi tidak valid karena fitur tersebut adalah pembentuk target.",
            ],
        }
    )

    st.dataframe(leakage_df, use_container_width=True, hide_index=True)

    callout(
        """
        <strong>Catatan penting:</strong>
        fitur <code>depression</code>, <code>anxiety</code>, dan <code>panic_attack</code>
        sengaja tidak digunakan dalam model valid karena ketiganya adalah pembentuk target risiko.
        Jika fitur tersebut dimasukkan, skor model dapat menjadi sangat tinggi, tetapi itu termasuk
        <strong>data leakage</strong> dan tidak valid sebagai evaluasi model.
        """
    )

    st.markdown("### Posisi Model dalam MindTrack UB")

    st.markdown(
        """
        Model Gradient Boosting digunakan sebagai **supporting evidence** bahwa pendekatan machine learning
        dapat membantu proses screening awal. Namun, karena dataset masih kecil dan belum memuat data kampus
        yang lebih kaya seperti presensi, LMS, keterlambatan tugas, partisipasi kegiatan, atau histori konseling,
        model tidak dijadikan satu-satunya dasar keputusan.

        Dalam rancangan akhir, MindTrack UB menggunakan kombinasi:

        - **dashboard monitoring** untuk membaca pola agregat,
        - **treatment gap analysis** untuk melihat mahasiswa berisiko yang belum mencari bantuan,
        - **model binary Gradient Boosting** untuk simulasi screening awal,
        - **validasi manusia/konselor** sebagai keputusan akhir.
        """
    )


# ============================================================
# TAB 6 - ML SCREENING SIMULATION
# ============================================================

with tab6:
    section_title(
        "🛟 ML Screening Simulation",
        "Simulasi ini menggunakan model Gradient Boosting untuk memprediksi No Risk vs At Risk berdasarkan fitur non-klinis.",
    )

    callout(
        """
        <strong>Disclaimer:</strong>
        simulasi ini menggunakan model machine learning, bukan rule-based scoring.
        Model hanya memakai fitur non-klinis seperti gender, age, course, year of study, CGPA, dan marital status.
        Output model berupa skor estimasi risiko terhadap kelas <strong>At Risk</strong>, lalu dashboard memetakannya menjadi <strong>No Risk</strong>, <strong>Need Monitoring</strong>, atau <strong>At Risk</strong>. Hasil ini bukan diagnosis medis.
        """
    )

    left, right = st.columns([0.95, 1.05])

    with left:
        st.markdown("### Input Simulasi Model")

        sim_gender = st.radio(
            "Gender",
            list(sorted(df["gender"].dropna().unique())),
            horizontal=True,
        )

        sim_age = st.slider(
            "Age",
            min_value=int(df["age"].min()),
            max_value=int(df["age"].max()),
            value=int(df["age"].median()),
        )

        sim_course = st.selectbox(
            "Course",
            list(sorted(df["course"].dropna().unique())),
        )

        sim_year = st.radio(
            "Year of Study",
            list(sort_years(df["year_of_study"].dropna().unique())),
            horizontal=True,
        )

        sim_cgpa = st.radio(
            "CGPA Range",
            [x for x in CGPA_ORDER if x in df["cgpa"].unique()],
        )

        sim_marital = st.radio(
            "Marital Status",
            list(sorted(df["marital_status"].dropna().unique())),
            horizontal=True,
        )

        st.markdown("---")
        st.caption(
            "Depression, anxiety, dan panic attack tidak dijadikan input model karena ketiganya adalah pembentuk target risiko."
        )

    with right:
        st.markdown("### Output Prediksi Model")

        input_data = pd.DataFrame(
            [
                {
                    "gender": sim_gender,
                    "age": sim_age,
                    "course": sim_course,
                    "year_of_study": sim_year,
                    "cgpa": sim_cgpa,
                    "marital_status": sim_marital,
                }
            ]
        )

        if hasattr(screening_model, "predict_proba"):
            proba = screening_model.predict_proba(input_data)[0]
            class_index = list(screening_model.classes_).index(1)
            at_risk_probability = float(proba[class_index])
        else:
            at_risk_probability = None

        if at_risk_probability < 0.50:
            pred_label = "No Risk"
            risk_class = "risk-low"
            color = "#22C55E"
            recommendation = [
                "Berikan edukasi mental health dan self-care.",
                "Tampilkan informasi umum layanan konseling kampus.",
                "Monitoring ringan dapat dilakukan melalui survei berkala.",
                "Tidak diperlukan prioritas outreach khusus berdasarkan output model saat ini.",
            ]

        elif at_risk_probability < 0.70:
            pred_label = "Need Monitoring"
            risk_class = "risk-medium"
            color = "#F59E0B"
            recommendation = [
                "Sarankan mahasiswa melakukan self-assessment lanjutan.",
                "Berikan informasi akses peer counselor atau konseling ringan.",
                "Lakukan monitoring berkala melalui survei singkat.",
                "Belum perlu dianggap prioritas tinggi tanpa validasi lanjutan.",
            ]

        else:
            pred_label = "At Risk"
            risk_class = "risk-high"
            color = "#EF4444"
            recommendation = [
                "Sarankan mahasiswa melakukan self-assessment lanjutan.",
                "Tampilkan opsi akses layanan konseling atau peer counselor.",
                "Berikan informasi pendampingan akademik bila risiko berkaitan dengan beban studi.",
                "Keputusan lanjutan tetap perlu diverifikasi oleh konselor atau pihak berwenang.",
            ]

        probability_text = (
            f"{at_risk_probability:.1%}"
            if at_risk_probability is not None
            else "Tidak tersedia"
        )

        st.markdown(
            f"""
            <div class="risk-box {risk_class}">
                <h3>{pred_label}</h3>
                <p><strong>Model:</strong> Gradient Boosting</p>
                <p><strong>Target model:</strong> Binary screening - No Risk vs At Risk</p>
                <p><strong>Skor Estimasi At Risk:</strong> {probability_text}</p>
                <p><strong>Kategori dashboard:</strong> {pred_label}</p>
                <p><strong>Input:</strong> {sim_gender}, {sim_age} tahun, {sim_course}, {sim_year}, CGPA {sim_cgpa}, marital status {sim_marital}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Rekomendasi Tindak Lanjut")
        for rec in recommendation:
            st.markdown(f"- {rec}")

        if at_risk_probability is not None:
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=at_risk_probability * 100,
                    title={
                        "text": "Estimated At-Risk Score",
                        "font": {"color": "#F8FAFC"},
                    },
                    number={
                        "suffix": "%",
                        "font": {"color": "#FFFFFF", "size": 42},
                    },
                    gauge={
                        "axis": {
                            "range": [0, 100],
                            "tickcolor": "#CBD5E1",
                        },
                        "bar": {"color": color},
                        "bgcolor": "rgba(15,23,42,0.5)",
                        "borderwidth": 1,
                        "bordercolor": "rgba(255,255,255,0.20)",
                        "steps": [
                            {"range": [0, 40], "color": "rgba(34,197,94,0.22)"},
                            {"range": [40, 70], "color": "rgba(245,158,11,0.22)"},
                            {"range": [70, 100], "color": "rgba(239,68,68,0.24)"},
                        ],
                    },
                )
            )
            fig = style_fig(fig, height=330)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Kenapa Simulasi Ini Bukan Rule-Based?")
    st.markdown(
        """
        Simulasi ini tidak menentukan risiko dari jumlah indikator depression, anxiety, dan panic attack.
        Sebaliknya, sistem melatih model Gradient Boosting dari data historis untuk mempelajari hubungan antara
        fitur non-klinis dengan status risiko binary.

        Alur simulasi:

        1. User memasukkan profil mahasiswa.
        2. Data input diproses oleh pipeline preprocessing.
        3. Model Gradient Boosting menghasilkan skor probabilitas terhadap kelas **At Risk**.
        4. Dashboard kemudian memetakan skor tersebut menjadi **No Risk**, **Need Monitoring**, atau **At Risk** agar hasilnya lebih mudah digunakan sebagai pendukung keputusan.

        Pendekatan ini lebih sesuai dengan hasil modeling terbaru karena simulation benar-benar berasal dari
        training model, bukan aturan manual.
        """
    )


# ============================================================
# TAB 7 - ETHICS & LIMITS
# ============================================================

with tab7:
    section_title("🔐 Ethics, Privacy, and Limitations")
    callout(
        """
        <strong>Prinsip utama:</strong>
        sistem tidak boleh digunakan untuk melabeli, menghukum, atau menstigma mahasiswa.
        Semua output harus dipakai sebagai sinyal awal untuk pendampingan yang aman, manusiawi,
        dan berbasis persetujuan.
        """,
        light=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Guardrails")
        st.markdown(
            """
            - Data sensitif harus dianonimkan atau dipseudonimkan.
            - Akses dashboard dibatasi untuk pihak berwenang.
            - Keputusan final tetap berada pada konselor/manusia.
            - Mahasiswa diberi informasi tentang tujuan penggunaan data.
            - Sistem tidak boleh menjadi alat hukuman akademik.
            """
        )

    with col2:
        st.markdown("### Limitasi Prototype")
        st.markdown(
            """
            - Dataset kecil dan bukan data mahasiswa UB secara langsung.
            - Belum memuat data presensi, LMS, keterlambatan tugas, dan histori konseling.
            - Model hanya proof of concept, bukan sistem produksi.
            - Implementasi nyata membutuhkan validasi etik dan klinis.
            - Risiko bias perlu diuji sebelum dipakai di lingkungan nyata.
            """
        )

    st.markdown("### Posisi Sistem dalam Smart Campus")
    st.markdown(
        """
        MindTrack UB diposisikan sebagai **decision support system**, bukan pengganti psikolog.
        Sistem membantu kampus membaca pola risiko secara agregat, menentukan prioritas awal,
        dan menghubungkan mahasiswa dengan layanan yang sesuai seperti edukasi mandiri,
        peer counselor, konseling ringan, atau rujukan profesional.
        """
    )