import os
import pandas as pd
import streamlit as st

from report import render_eda, render_bivariate_analysis, render_multivariate_analysis
from etl_prefect import run_etl
from eda_validation import main as run_validation

st.set_page_config(
    page_title="Healthcare Reporting & Data Quality Dashboard",
    layout="wide"
)

CURATED_PATH = "data/curated"
REPORTS_PATH = "reports"

PATIENTS_PATH = os.path.join(CURATED_PATH, "patients.csv")
VISITS_PATH = os.path.join(CURATED_PATH, "visits.csv")
MEASURES_PATH = os.path.join(CURATED_PATH, "measures.csv")


# =========================
# 🎨 Clean Teal Theme (NO vertical bars)
# =========================
def apply_custom_theme():
    st.markdown(
        """
        <style>
        :root {
            --teal-primary: #0f766e;
            --teal-secondary: #14b8a6;
            --teal-light: #ccfbf1;
            --teal-dark: #134e4a;
            --bg-main: #ecfdf5;
            --bg-card: #ffffff;
        }

        .main {
            background-color: var(--bg-main);
        }

        .block-container {
            padding-top: 2rem;
        }

        h1 {
            color: var(--teal-dark) !important;
            font-weight: 700;
        }

        h2 {
            color: var(--teal-primary) !important;
            margin-top: 20px;
            font-weight: 600;
        }

        h3 {
            color: var(--teal-dark) !important;
        }

        div[data-testid="stMetric"] {
            background-color: var(--bg-card);
            border: 1px solid #99f6e4;
            padding: 18px;
            border-radius: 14px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        }

        section[data-testid="stSidebar"] {
            background-color: #d1fae5;
        }

        button[data-baseweb="tab"] {
            font-size: 15px;
            font-weight: 600;
            color: var(--teal-dark);
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            border-bottom: 3px solid var(--teal-primary);
            color: var(--teal-primary);
        }
        </style>
        """,
        unsafe_allow_html=True
    )


# =========================
# 🛠 Ensure Data Exists (Cloud-safe)
# =========================
def ensure_project_files():
    required_files = [PATIENTS_PATH, VISITS_PATH, MEASURES_PATH]
    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        os.makedirs(CURATED_PATH, exist_ok=True)
        os.makedirs(REPORTS_PATH, exist_ok=True)

        run_etl()
        run_validation()


# =========================
# 📥 Load Data
# =========================
@st.cache_data
def load_data():
    ensure_project_files()

    patients = pd.read_csv(PATIENTS_PATH)
    visits = pd.read_csv(VISITS_PATH, parse_dates=["visit_date"])
    measures = pd.read_csv(MEASURES_PATH)

    return patients, visits, measures


# =========================
# 📊 Dashboard
# =========================
def render_main_dashboard():
    patients, visits, measures = load_data()
    full_data = patients.merge(measures, on="patient_id", how="left")

    st.sidebar.header("Filters")

    selected_conditions = st.sidebar.multiselect(
        "Condition",
        options=sorted(full_data["condition"].unique()),
        default=sorted(full_data["condition"].unique())
    )

    selected_departments = st.sidebar.multiselect(
        "Department",
        options=sorted(visits["department"].unique()),
        default=sorted(visits["department"].unique())
    )

    filtered_patients = full_data[full_data["condition"].isin(selected_conditions)]
    filtered_visits = visits[
        (visits["department"].isin(selected_departments)) &
        (visits["patient_id"].isin(filtered_patients["patient_id"]))
    ]

    total_patients = filtered_patients["patient_id"].nunique()
    total_visits = len(filtered_visits)
    no_show_rate = filtered_visits["no_show_flag"].mean() if total_visits > 0 else 0
    department_count = filtered_visits["department"].nunique()

    diabetic = filtered_patients[filtered_patients["condition"] == "diabetes"]
    hyper = filtered_patients[filtered_patients["condition"] == "hypertension"]

    hba1c = (diabetic["hba1c_test_done"] == "yes").mean() * 100 if len(diabetic) else 0
    annual = (filtered_patients["annual_checkup"] == "yes").mean() * 100 if len(filtered_patients) else 0
    bp = (hyper["bp_controlled"] == "yes").mean() * 100 if len(hyper) else 0

    st.markdown("<h2>Operational Metrics</h2>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Patients", total_patients)
    c2.metric("Total Visits", total_visits)
    c3.metric("No-Show Rate", f"{no_show_rate:.1%}")
    c4.metric("Departments", department_count)

    st.markdown("<h2>Patient Visit Overview</h2>", unsafe_allow_html=True)

    if not filtered_visits.empty:
        st.bar_chart(filtered_visits["department"].value_counts())

        visits_trend = filtered_visits.copy()
        visits_trend["visit_week"] = visits_trend["visit_date"].dt.to_period("W").astype(str)
        st.line_chart(visits_trend.groupby("visit_week").size())
    else:
        st.info("No visit data available for selected filters.")

    st.markdown("<h2>HEDIS Metrics</h2>", unsafe_allow_html=True)

    q1, q2, q3 = st.columns(3)
    q1.metric("HbA1c Compliance", f"{hba1c:.1f}%")
    q2.metric("Annual Checkup", f"{annual:.1f}%")
    q3.metric("BP Control", f"{bp:.1f}%")


# =========================
# 🚀 App Entry
# =========================
def main():
    apply_custom_theme()

    st.markdown("<h1>Healthcare Reporting & Data Quality Dashboard</h1>", unsafe_allow_html=True)

    st.markdown(
        "<p>A modular healthcare analytics dashboard with operational metrics, EDA, and advanced analysis.</p>",
        unsafe_allow_html=True
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dashboard",
        "📈 EDA",
        "📉 Bivariate",
        "🔗 Multivariate"
    ])

    with tab1:
        render_main_dashboard()

    with tab2:
        ensure_project_files()
        render_eda()

    with tab3:
        ensure_project_files()
        render_bivariate_analysis()

    with tab4:
        ensure_project_files()
        render_multivariate_analysis()


if __name__ == "__main__":
    main()