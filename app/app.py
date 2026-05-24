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
from sklearn.linear_model import LogisticRegression

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler


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
        max-width: 980px;
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
# 3. PATH CONFIG
# ============================================================

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
PIPELINE_OUTPUT_DIR = PROJECT_ROOT / "prediction_pipeline_outputs"
TABLE_DIR = PIPELINE_OUTPUT_DIR / "tables"
VIZ_DIR = PIPELINE_OUTPUT_DIR / "visualizations"


# ============================================================
# 4. DATA LOADING & CLEANING
# ============================================================

DATA_CANDIDATES = [
    DATA_DIR / "Student Mental health.csv",
    DATA_DIR / "student_mental_health.csv",
    TABLE_DIR / "student_mental_health_cleaned.csv",
    TABLE_DIR / "processed_student_mental_health.csv",
    PROJECT_ROOT / "outputs" / "processed_student_mental_health.csv",
    PROJECT_ROOT / "Student Mental health.csv",
    PROJECT_ROOT / "student_mental_health.csv",
]


def find_dataset_path():
    for path in DATA_CANDIDATES:
        if path.exists():
            return path
    return None


def normalize_column_name(col):
    return (
        str(col)
        .strip()
        .lower()
        .replace("?", "")
        .replace("/", "_")
        .replace("-", "_")
        .replace(" ", "_")
        .replace("__", "_")
    )


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        raw = pd.read_csv(uploaded_file)
        source_path = "uploaded file"
    else:
        dataset_path = find_dataset_path()
        if dataset_path is None:
            return None, None
        raw = pd.read_csv(dataset_path)
        source_path = str(dataset_path)

    df = raw.copy()
    df.columns = [normalize_column_name(col) for col in df.columns]

    alias_map = {
        "gender": ["choose_your_gender", "gender"],
        "age": ["age"],
        "course": ["what_is_your_course", "course", "course_clean"],
        "year_of_study": [
            "your_current_year_of_study",
            "year_of_study",
            "year_clean",
            "study_year",
            "year",
            "current_year",
        ],
        "cgpa": ["what_is_your_cgpa", "cgpa", "cgpa_range"],
        "marital_status": ["marital_status", "married_status"],
        "depression": ["do_you_have_depression", "depression", "depression_indicator"],
        "anxiety": ["do_you_have_anxiety", "anxiety", "anxiety_indicator"],
        "panic_attack": ["do_you_have_panic_attack", "panic_attack", "panic", "panic_attack_indicator"],
        "seek_treatment": [
            "did_you_seek_any_specialist_for_a_treatment",
            "seek_treatment",
            "seek_specialist",
            "sought_treatment",
            "specialist_treatment",
            "treatment",
        ],
        "depression_flag": ["depression_flag", "has_depression"],
        "anxiety_flag": ["anxiety_flag", "has_anxiety"],
        "panic_flag": ["panic_flag", "panic_attack_flag", "has_panic_attack"],
        "support_status_existing": ["support_status", "treatment_status", "help_seeking_status"],
    }

    rename_dict = {}
    for canonical, aliases in alias_map.items():
        if canonical in df.columns:
            continue
        for alias in aliases:
            if alias in df.columns:
                rename_dict[alias] = canonical
                break

    df = df.rename(columns=rename_dict)

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
        if pd.isna(value):
            return "No"

        text = str(value).strip().lower()

        yes_values = [
            "yes",
            "y",
            "true",
            "1",
            "1.0",
            "reached support",
            "sought support",
            "seek treatment",
            "sought treatment",
            "has indicator",
            "at risk",
        ]

        no_values = [
            "no",
            "n",
            "false",
            "0",
            "0.0",
            "no support",
            "silent struggle",
            "no risk",
            "no indicator",
        ]

        if text in yes_values:
            return "Yes"
        if text in no_values:
            return "No"

        if "reached" in text or "sought" in text or "seek" in text:
            return "Yes"
        if "silent" in text or "no support" in text:
            return "No"

        return "No"

    if "support_status_existing" in df.columns:
        seek_from_support = df["support_status_existing"].apply(clean_yes_no)
        raw_seek = df["seek_treatment"].apply(clean_yes_no)

        if (raw_seek == "Yes").sum() == 0 and (seek_from_support == "Yes").sum() > 0:
            df["seek_treatment"] = seek_from_support
        else:
            df["seek_treatment"] = raw_seek
    else:
        df["seek_treatment"] = df["seek_treatment"].apply(clean_yes_no)

    for col in ["depression", "anxiety", "panic_attack"]:
        df[col] = df[col].apply(clean_yes_no)

    def get_flag(text_col, flag_col):
        if flag_col in df.columns:
            flag = pd.to_numeric(df[flag_col], errors="coerce")
            if flag.notna().sum() > 0:
                return flag.fillna(0).astype(int)
        return (df[text_col] == "Yes").astype(int)

    df["depression_flag"] = get_flag("depression", "depression_flag")
    df["anxiety_flag"] = get_flag("anxiety", "anxiety_flag")
    df["panic_flag"] = get_flag("panic_attack", "panic_flag")

    df["depression"] = np.where(df["depression_flag"] == 1, "Yes", "No")
    df["anxiety"] = np.where(df["anxiety_flag"] == 1, "Yes", "No")
    df["panic_attack"] = np.where(df["panic_flag"] == 1, "Yes", "No")

    df["symptom_count"] = df[["depression_flag", "anxiety_flag", "panic_flag"]].sum(axis=1)

    def risk_label(score):
        if score == 0:
            return "Low Risk"
        if score == 1:
            return "Medium Risk"
        return "High Risk"

    df["risk_level"] = df["symptom_count"].apply(risk_label)
    df["risk_score"] = df["symptom_count"] / 3
    df["binary_risk"] = np.where(df["symptom_count"] > 0, "At Risk", "No Risk")

    def support_status_detail(row):
        if row["symptom_count"] == 0:
            return "No Indicator"
        if row["seek_treatment"] == "Yes":
            return "Reached Support"
        return "Silent Struggle"

    df["support_status"] = df.apply(support_status_detail, axis=1)

    df["silent_struggle"] = np.where(
        (df["symptom_count"] > 0) & (df["seek_treatment"] == "No"),
        "Silent Struggle",
        "Other",
    )

    def priority_segment(row):
        if row["symptom_count"] == 0:
            return "Low Priority"
        if row["symptom_count"] >= 2 and row["seek_treatment"] == "No":
            return "High Priority"
        if row["symptom_count"] == 1 and row["seek_treatment"] == "No":
            return "Need Monitoring"
        if row["symptom_count"] > 0 and row["seek_treatment"] == "Yes":
            return "Reached Support"
        return "Low Priority"

    df["priority_segment"] = df.apply(priority_segment, axis=1)

    return df, source_path


df, data_source = load_data()

if df is None:
    st.error("Dataset tidak ditemukan. Upload file `Student Mental health.csv` dulu untuk menjalankan dashboard.")
    uploaded_file = st.file_uploader("Upload dataset CSV", type=["csv"])
    if uploaded_file is not None:
        df, data_source = load_data(uploaded_file)
    else:
        st.stop()


# ============================================================
# 5. MODEL TRAINING FOR SIMULATION
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
            ("onehot", make_one_hot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )

    pipeline = ImbPipeline(
        steps=[
            ("preprocess", preprocessor),
            ("sampler", RandomOverSampler(random_state=42)),
            ("model", model),
        ]
    )

    pipeline.fit(X, y)

    return pipeline, feature_cols


screening_model, screening_feature_cols = train_screening_model(df)


# ============================================================
# 6. CONSTANTS & HELPER FUNCTIONS
# ============================================================

RISK_ORDER = ["Low Risk", "Medium Risk", "High Risk"]
RISK_COLORS = {
    "Low Risk": "#22C55E",
    "Medium Risk": "#F59E0B",
    "High Risk": "#EF4444",
}

SUPPORT_COLORS = {
    "No Indicator": "#64748B",
    "Silent Struggle": "#F59E0B",
    "Reached Support": "#22C55E",
    "Other": "#64748B",
}

PRIORITY_ORDER = ["Low Priority", "Need Monitoring", "High Priority", "Reached Support"]
PRIORITY_COLORS = {
    "Low Priority": "#22C55E",
    "Need Monitoring": "#F59E0B",
    "High Priority": "#EF4444",
    "Reached Support": "#06B6D4",
}

INDICATOR_COLORS = {
    "Depression": "#A78BFA",
    "Anxiety": "#22D3EE",
    "Panic Attack": "#FB7185",
}

CGPA_ORDER = [
    "0 - 1.99",
    "2.00 - 2.49",
    "2.50 - 2.99",
    "3.00 - 3.49",
    "3.50 - 4.00",
    "Unknown",
]


def safe_pct(part, total):
    if total == 0:
        return 0
    return round((part / total) * 100, 1)


def sort_years(values):
    def key_func(x):
        digits = "".join([c for c in str(x) if c.isdigit()])
        return int(digits) if digits else 99

    return sorted(values, key=key_func)


def style_fig(fig, height=430, showlegend=True):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F8FAFC", family="Inter, sans-serif", size=13),
        margin=dict(l=35, r=35, t=95, b=95),
        title=dict(
            x=0.02,
            xanchor="left",
            y=0.97,
            yanchor="top",
            font=dict(size=17, color="#FFFFFF"),
        ),
        legend=dict(
            title="",
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#F8FAFC", size=12),
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5,
        ),
        showlegend=showlegend,
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.12)",
        zerolinecolor="rgba(148,163,184,0.18)",
        tickfont=dict(color="#CBD5E1"),
        title_font=dict(color="#F8FAFC"),
        automargin=True,
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.12)",
        zerolinecolor="rgba(148,163,184,0.18)",
        tickfont=dict(color="#CBD5E1"),
        title_font=dict(color="#F8FAFC"),
        automargin=True,
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
    priority_options = PRIORITY_ORDER

    selected_gender = pill_filter("Gender", gender_options, "filter_gender")
    selected_year = pill_filter("Year of Study", year_options, "filter_year")
    selected_cgpa = pill_filter("CGPA Range", cgpa_options, "filter_cgpa")
    selected_priority = pill_filter("Priority Segment", priority_options, "filter_priority")

    filtered = data[
        data["gender"].isin(selected_gender)
        & data["year_of_study"].isin(selected_year)
        & data["cgpa"].isin(selected_cgpa)
        & data["priority_segment"].isin(selected_priority)
    ].copy()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📌 Catatan")
    st.sidebar.info(
        "Dashboard ini adalah prototype berbasis dataset kecil. Output dipakai sebagai sinyal awal, bukan diagnosis medis."
    )

    with st.sidebar.expander("Data source"):
        st.caption(str(data_source))

    return filtered


filtered_df = filter_dataframe(df)


# ============================================================
# 7. HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>MindTrack UB</h1>
        <p>
            Prototype decision support system untuk membaca Silent Struggle, Support Gap, dan prioritas
            pendampingan mahasiswa berbasis analisis data serta simulasi machine learning.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if filtered_df.empty:
    st.warning("Tidak ada data yang sesuai dengan filter saat ini. Pilih kembali minimal satu opsi pada setiap filter.")
    st.stop()


# ============================================================
# 8. SUMMARY METRICS
# ============================================================

total_students = len(filtered_df)
at_risk_count = int((filtered_df["symptom_count"] > 0).sum())
silent_count = int(((filtered_df["symptom_count"] > 0) & (filtered_df["seek_treatment"] == "No")).sum())
reached_count = int(((filtered_df["symptom_count"] > 0) & (filtered_df["seek_treatment"] == "Yes")).sum())
high_priority_count = int((filtered_df["priority_segment"] == "High Priority").sum())
support_gap_rate = safe_pct(silent_count, at_risk_count)

st.markdown("<br>", unsafe_allow_html=True)

m1, m2, m3, m4, m5 = st.columns(5)

with m1:
    metric_card("Total Responden", total_students, "Jumlah data setelah filter aktif.", "#8B5CF6")
with m2:
    metric_card("At Risk", at_risk_count, f"{safe_pct(at_risk_count, total_students)}% memiliki minimal 1 indikator.", "#06B6D4")
with m3:
    metric_card("Silent Struggle", silent_count, "Berisiko tetapi belum mencari bantuan.", "#F59E0B")
with m4:
    metric_card("Reached Support", reached_count, "Berisiko dan sudah mencari bantuan spesialis.", "#22C55E")
with m5:
    metric_card("Support Gap", f"{support_gap_rate}%", "Persentase silent struggle dari kelompok At Risk.", "#EF4444")


# ============================================================
# 9. TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "📌 Executive Overview",
        "🎯 Risk & Priority",
        "🧩 Support Gap",
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
        masalah utama bukan hanya jumlah mahasiswa yang memiliki indikator risiko,
        tetapi adanya <strong>Silent Struggle</strong>, yaitu mahasiswa yang sudah menunjukkan indikator
        mental health namun belum mencari bantuan spesialis.
        """
    )

    col1, col2 = st.columns([1.1, 1])

    with col1:
        indicator_df = pd.DataFrame(
            {
                "Indicator": ["Depression", "Anxiety", "Panic Attack"],
                "Count": [
                    int((filtered_df["depression"] == "Yes").sum()),
                    int((filtered_df["anxiety"] == "Yes").sum()),
                    int((filtered_df["panic_attack"] == "Yes").sum()),
                ],
            }
        )

        fig = px.bar(
            indicator_df,
            x="Indicator",
            y="Count",
            text="Count",
            color="Indicator",
            color_discrete_map=INDICATOR_COLORS,
            title="Prevalensi Indikator Mental Health",
        )
        fig.update_traces(textposition="outside", marker_line_width=0)
        fig = style_fig(fig, height=430)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        support_df = pd.DataFrame(
            {
                "Status": ["Silent Struggle", "Reached Support"],
                "Count": [silent_count, reached_count],
            }
        )

        fig = px.pie(
            support_df,
            names="Status",
            values="Count",
            hole=0.58,
            color="Status",
            color_discrete_map={
                "Silent Struggle": "#F59E0B",
                "Reached Support": "#22C55E",
            },
            title="Komposisi Support Gap pada Kelompok At Risk",
        )
        fig.update_traces(
            textinfo="percent+label",
            textfont=dict(color="#FFFFFF", size=13),
            marker=dict(line=dict(color="rgba(255,255,255,0.18)", width=1)),
        )
        fig = style_fig(fig, height=430)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Insight Ringkas")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="mini-card">
                <h4>1. Risiko tidak selalu terlihat langsung</h4>
                <p>
                    Indikator depression, anxiety, dan panic attack digunakan untuk membaca
                    kondisi awal responden, bukan untuk menetapkan diagnosis medis.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="mini-card">
                <h4>2. Silent Struggle adalah masalah utama</h4>
                <p>
                    Ada mahasiswa yang sudah menunjukkan indikator risiko, tetapi belum mencari bantuan.
                    Inilah alasan sistem screening dan dashboard monitoring menjadi relevan.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            """
            <div class="mini-card">
                <h4>3. Dashboard mendukung prioritisasi</h4>
                <p>
                    Kampus dapat membaca pola agregat dan menentukan prioritas pendampingan,
                    sementara keputusan akhir tetap berada pada konselor atau pihak berwenang.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# TAB 2 - RISK & PRIORITY
# ============================================================

with tab2:
    section_title(
        "🎯 Risk & Priority Segmentation",
        "Segmentasi ini membaca kombinasi jumlah indikator mental health dan status pencarian bantuan.",
    )

    callout(
        """
        <strong>Definisi analitik:</strong>
        Low Priority = tidak ada indikator, Need Monitoring = 1 indikator dan belum mencari bantuan,
        High Priority = 2–3 indikator dan belum mencari bantuan, Reached Support = berisiko dan sudah mencari bantuan.
        """
    )

    risk_summary = (
        filtered_df.groupby("risk_level")
        .agg(
            total=("risk_level", "count"),
            avg_symptom=("symptom_count", "mean"),
            reached_support=("seek_treatment", lambda x: (x == "Yes").sum()),
        )
        .reindex(RISK_ORDER)
        .reset_index()
        .fillna(0)
    )

    risk_summary["avg_symptom"] = risk_summary["avg_symptom"].round(2)
    risk_summary["support_rate"] = risk_summary.apply(
        lambda row: safe_pct(row["reached_support"], row["total"]), axis=1
    )

    priority_summary = (
        filtered_df["priority_segment"]
        .value_counts()
        .reindex(PRIORITY_ORDER, fill_value=0)
        .reset_index()
    )
    priority_summary.columns = ["Priority Segment", "Count"]

    col1, col2 = st.columns([1.1, 1])

    with col1:
        fig = px.bar(
            priority_summary,
            x="Priority Segment",
            y="Count",
            text="Count",
            color="Priority Segment",
            color_discrete_map=PRIORITY_COLORS,
            title="Jumlah Mahasiswa per Priority Segment",
        )
        fig.update_traces(textposition="outside")
        fig = style_fig(fig, height=470, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            risk_summary,
            x="risk_level",
            y="support_rate",
            text="support_rate",
            color="risk_level",
            color_discrete_map=RISK_COLORS,
            title="Support Rate per Risk Level (%)",
            labels={"risk_level": "Risk Level", "support_rate": "Support Rate (%)"},
        )
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig = style_fig(fig, height=470, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Tabel Ringkasan Risk Level")
    st.dataframe(
        risk_summary.rename(
            columns={
                "risk_level": "Risk Level",
                "total": "Total",
                "avg_symptom": "Avg Symptom Count",
                "reached_support": "Reached Support",
                "support_rate": "Support Rate (%)",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# TAB 3 - SUPPORT GAP
# ============================================================

with tab3:
    section_title("🧩 Support Gap & Silent Struggle Analysis")
    callout(
        """
        <strong>Support gap</strong> terjadi ketika mahasiswa sudah memiliki indikator risiko,
        tetapi belum mencari bantuan spesialis. Dalam dashboard ini, kondisi tersebut disebut
        <strong>Silent Struggle</strong>.
        """
    )

    support_by_risk = (
        filtered_df.groupby(["risk_level", "support_status"])
        .size()
        .reset_index(name="count")
    )

    fig = px.bar(
        support_by_risk,
        x="risk_level",
        y="count",
        color="support_status",
        barmode="group",
        category_orders={
            "risk_level": RISK_ORDER,
            "support_status": ["No Indicator", "Silent Struggle", "Reached Support"],
        },
        color_discrete_map=SUPPORT_COLORS,
        title="Support Status per Risk Level",
        labels={
            "risk_level": "Risk Level",
            "count": "Jumlah Mahasiswa",
            "support_status": "Support Status",
        },
        text="count",
    )
    fig.update_traces(textposition="outside")
    fig = style_fig(fig, height=500)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card(
            "At Risk",
            at_risk_count,
            "Memiliki minimal satu indikator mental health.",
            "#06B6D4",
        )

    with c2:
        metric_card(
            "Silent Struggle",
            silent_count,
            f"{support_gap_rate}% dari kelompok At Risk belum mencari bantuan.",
            "#F59E0B",
        )

    with c3:
        metric_card(
            "High Priority",
            high_priority_count,
            "Memiliki 2–3 indikator dan belum mencari bantuan.",
            "#EF4444",
        )

    st.markdown("### Interpretasi")
    st.markdown(
        """
        Support gap menunjukkan bahwa keberadaan layanan saja belum otomatis membuat mahasiswa mencari bantuan.
        Sistem pendukung keputusan dapat membantu kampus melakukan pendekatan yang lebih proaktif,
        misalnya melalui edukasi, self-assessment lanjutan, peer counselor, konseling ringan,
        atau prioritas outreach oleh pihak berwenang.
        """
    )


# ============================================================
# TAB 4 - PATTERN ANALYSIS
# ============================================================

with tab4:
    section_title(
        "📊 Pattern Analysis",
        "Bagian ini melihat pola Silent Struggle dan priority segment berdasarkan atribut demografis dan akademik.",
    )

    col1, col2 = st.columns(2)

    with col1:
        gender_priority = (
            filtered_df.groupby(["gender", "priority_segment"])
            .size()
            .reset_index(name="count")
        )

        fig = px.bar(
            gender_priority,
            x="gender",
            y="count",
            color="priority_segment",
            barmode="group",
            category_orders={"priority_segment": PRIORITY_ORDER},
            color_discrete_map=PRIORITY_COLORS,
            title="Priority Segment berdasarkan Gender",
            labels={"gender": "Gender", "count": "Jumlah", "priority_segment": "Priority Segment"},
            text="count",
        )
        fig.update_traces(textposition="outside")
        fig = style_fig(fig, height=520)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        year_priority = (
            filtered_df.groupby(["year_of_study", "priority_segment"])
            .size()
            .reset_index(name="count")
        )

        fig = px.bar(
            year_priority,
            x="year_of_study",
            y="count",
            color="priority_segment",
            barmode="group",
            category_orders={
                "priority_segment": PRIORITY_ORDER,
                "year_of_study": sort_years(filtered_df["year_of_study"].unique()),
            },
            color_discrete_map=PRIORITY_COLORS,
            title="Priority Segment berdasarkan Tahun Studi",
            labels={"year_of_study": "Year of Study", "count": "Jumlah", "priority_segment": "Priority Segment"},
            text="count",
        )
        fig.update_traces(textposition="outside")
        fig = style_fig(fig, height=520)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        cgpa_priority = (
            filtered_df.groupby(["cgpa", "priority_segment"])
            .size()
            .reset_index(name="count")
        )

        fig = px.bar(
            cgpa_priority,
            x="cgpa",
            y="count",
            color="priority_segment",
            barmode="stack",
            category_orders={"priority_segment": PRIORITY_ORDER, "cgpa": CGPA_ORDER},
            color_discrete_map=PRIORITY_COLORS,
            title="Priority Segment berdasarkan CGPA",
            labels={"cgpa": "CGPA Range", "count": "Jumlah", "priority_segment": "Priority Segment"},
        )
        fig = style_fig(fig, height=520)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        course_silent = (
            filtered_df[filtered_df["silent_struggle"] == "Silent Struggle"]
            .groupby("course")
            .size()
            .sort_values(ascending=False)
            .head(10)
            .reset_index(name="count")
        )

        if course_silent.empty:
            st.info("Tidak ada data Silent Struggle pada filter saat ini.")
        else:
            fig = px.bar(
                course_silent,
                x="count",
                y="course",
                orientation="h",
                title="Top 10 Course dengan Silent Struggle",
                labels={"count": "Jumlah", "course": "Course"},
                text="count",
                color="count",
                color_continuous_scale="Turbo",
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(coloraxis_showscale=False)
            fig = style_fig(fig, height=520, showlegend=False)
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
        pendekatan valid terbaik berdasarkan F1 Macro adalah Binary Classification dengan target
        <strong>No Risk vs At Risk</strong> menggunakan <strong>Logistic Regression Balanced + RandomOverSampler</strong>.
        Model ini dipakai sebagai proof of concept untuk screening awal, bukan diagnosis medis.
        """
    )

    st.markdown("### Best Valid Model")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "Best Model",
            "LogReg + ROS",
            "Model terbaik berdasarkan F1 Macro.",
            "#8B5CF6",
        )

    with c2:
        metric_card(
            "Accuracy CV",
            "0.662",
            "Rata-rata accuracy cross-validation.",
            "#06B6D4",
        )

    with c3:
        metric_card(
            "Balanced Acc.",
            "0.635",
            "Lebih adil untuk data tidak seimbang.",
            "#F59E0B",
        )

    with c4:
        metric_card(
            "F1 Macro",
            "0.633",
            "Metrik utama untuk evaluasi model.",
            "#22C55E",
        )

    model_result_df = pd.DataFrame(
        {
            "Experiment": [
                "Binary At Risk with Oversampling",
                "Binary At Risk",
                "Binary At Risk with Oversampling",
                "Binary At Risk",
                "Binary At Risk",
                "Multiclass Risk Level",
                "Dummy Baseline",
            ],
            "Model": [
                "LogReg Balanced + ROS",
                "Gradient Boosting",
                "RF + ROS",
                "Logistic Regression",
                "Decision Tree Balanced",
                "Extra Trees Balanced",
                "Dummy Most Frequent",
            ],
            "Accuracy Mean": [0.662, 0.693, 0.613, 0.683, 0.573, 0.515, 0.634],
            "Balanced Accuracy Mean": [0.635, 0.627, 0.607, 0.600, 0.610, 0.517, 0.500],
            "F1 Macro Mean": [0.633, 0.618, 0.596, 0.576, 0.570, 0.516, 0.388],
            "F1 Weighted Mean": [0.650, 0.660, 0.615, 0.634, 0.572, 0.507, 0.492],
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
    fig = style_fig(fig, height=520)
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
                    dengan jumlah data yang terbatas. Akibatnya, performa model cenderung tidak stabil.
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
                    Untuk early warning, pertanyaan awal yang paling penting adalah apakah mahasiswa
                    masuk kelompok berisiko atau tidak. Detail prioritas tetap dibaca dari analisis dashboard.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Valid Modeling vs Leakage Modeling")

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
        <strong>data leakage</strong>.
        """
    )


# ============================================================
# TAB 6 - ML SCREENING SIMULATION
# ============================================================

with tab6:
    section_title(
        "🛟 ML Screening Simulation",
        "Simulasi ini menggunakan model LogReg Balanced + RandomOverSampler untuk menghasilkan skor estimasi At Risk.",
    )

    callout(
        """
        <strong>Disclaimer:</strong>
        simulasi ini menggunakan model machine learning, bukan rule-based scoring.
        Model hanya memakai fitur non-klinis seperti gender, age, course, year of study, CGPA, dan marital status.
        Output model berupa skor estimasi terhadap kelas <strong>At Risk</strong>, lalu dashboard memetakannya
        menjadi <strong>No Risk</strong>, <strong>Need Monitoring</strong>, atau <strong>At Risk</strong>.
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

        proba = screening_model.predict_proba(input_data)[0]
        class_index = list(screening_model.classes_).index(1)
        at_risk_probability = float(proba[class_index])

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

        probability_text = f"{at_risk_probability:.1%}"

        st.markdown(
            f"""
            <div class="risk-box {risk_class}">
                <h3>{pred_label}</h3>
                <p><strong>Model:</strong> Logistic Regression Balanced + RandomOverSampler</p>
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
                        {"range": [0, 50], "color": "rgba(34,197,94,0.22)"},
                        {"range": [50, 70], "color": "rgba(245,158,11,0.22)"},
                        {"range": [70, 100], "color": "rgba(239,68,68,0.24)"},
                    ],
                },
            )
        )
        fig = style_fig(fig, height=360, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Interpretasi Simulasi")
    st.markdown(
        """
        Model menghasilkan skor probabilitas terhadap kelas **At Risk** berdasarkan pola pada data historis.
        Dashboard kemudian memetakan skor tersebut menjadi:

        - **No Risk** jika skor < 50%
        - **Need Monitoring** jika skor 50% sampai 69.9%
        - **At Risk** jika skor >= 70%

        Kategori ini digunakan sebagai dukungan awal untuk pengambilan keputusan, bukan diagnosis medis.
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
        Sistem membantu kampus membaca pola risiko secara agregat, melihat Support Gap,
        mengidentifikasi Silent Struggle, dan menghubungkan mahasiswa dengan layanan yang sesuai
        seperti edukasi mandiri, peer counselor, konseling ringan, atau rujukan profesional.
        """
    )