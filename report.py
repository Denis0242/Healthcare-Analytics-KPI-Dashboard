import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

REPORTS_PATH = "reports"
VALIDATION_SUMMARY_PATH = os.path.join(REPORTS_PATH, "validation_summary.csv")
ANALYSIS_DATASET_PATH = os.path.join(REPORTS_PATH, "analysis_dataset.csv")


def _load_reports():
    validation_df = None
    analysis_df = None

    if os.path.exists(VALIDATION_SUMMARY_PATH):
        validation_df = pd.read_csv(VALIDATION_SUMMARY_PATH)

    if os.path.exists(ANALYSIS_DATASET_PATH):
        analysis_df = pd.read_csv(ANALYSIS_DATASET_PATH)

    return validation_df, analysis_df


def render_eda() -> None:
    st.markdown("<h2 style='color:#0f766e;'>EDA Overview</h2>", unsafe_allow_html=True)

    validation_df, analysis_df = _load_reports()

    if validation_df is None or analysis_df is None:
        st.info("Run `uv run eda_validation.py` first to generate report files.")
        return

    st.markdown("<h3 style='color:#0f766e;'>Validation Summary</h3>", unsafe_allow_html=True)
    st.dataframe(validation_df, use_container_width=True)

    st.markdown("<h3 style='color:#0f766e;'>Data Quality Snapshot</h3>", unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)
    d1.metric(
        "Patient Duplicates",
        int(validation_df.loc[validation_df["table_name"] == "patients", "duplicate_rows"].iloc[0])
    )
    d2.metric(
        "Visit Duplicates",
        int(validation_df.loc[validation_df["table_name"] == "visits", "duplicate_rows"].iloc[0])
    )
    d3.metric(
        "Measure Duplicates",
        int(validation_df.loc[validation_df["table_name"] == "measures", "duplicate_rows"].iloc[0])
    )

    st.markdown("<h3 style='color:#0f766e;'>Dataset Snapshot</h3>", unsafe_allow_html=True)
    st.dataframe(analysis_df.head(20), use_container_width=True)

    st.markdown("<h3 style='color:#0f766e;'>Condition Distribution</h3>", unsafe_allow_html=True)
    condition_dist = analysis_df["condition"].value_counts()
    st.bar_chart(condition_dist)

    st.markdown("<h3 style='color:#0f766e;'>Age Distribution by Condition</h3>", unsafe_allow_html=True)
    age_summary = (
        analysis_df.groupby("condition")["age"]
        .agg(["mean", "min", "max"])
        .round(1)
    )
    st.dataframe(age_summary, use_container_width=True)

    st.markdown("<h3 style='color:#0f766e;'>Total Visits Distribution</h3>", unsafe_allow_html=True)
    visit_dist = analysis_df["total_visits"].value_counts().sort_index()
    st.bar_chart(visit_dist)


def render_bivariate_analysis() -> None:
    st.markdown("<h2 style='color:#0f766e;'>Bivariate Analysis</h2>", unsafe_allow_html=True)

    _, analysis_df = _load_reports()
    if analysis_df is None:
        st.info("Run `uv run eda_validation.py` first to generate the analysis dataset.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<h3 style='color:#0f766e;'>Average Visits by Condition</h3>", unsafe_allow_html=True)
        avg_visits = (
            analysis_df.groupby("condition")["total_visits"]
            .mean()
            .round(2)
            .sort_values(ascending=False)
        )
        st.bar_chart(avg_visits)

        st.markdown("<h3 style='color:#0f766e;'>Average No-Show Rate by Condition</h3>", unsafe_allow_html=True)
        no_show_by_condition = (
            analysis_df.groupby("condition")["no_show_rate"]
            .mean()
            .round(3)
            .sort_values(ascending=False)
        )
        st.bar_chart(no_show_by_condition)

    with col2:
        st.markdown("<h3 style='color:#0f766e;'>HbA1c Completion by Condition</h3>", unsafe_allow_html=True)
        hba1c_by_condition = (
            analysis_df.groupby("condition")["hba1c_completed_flag"]
            .mean()
            .mul(100)
            .round(1)
            .sort_values(ascending=False)
        )
        st.bar_chart(hba1c_by_condition)

        st.markdown("<h3 style='color:#0f766e;'>BP Control by Condition</h3>", unsafe_allow_html=True)
        bp_by_condition = (
            analysis_df.groupby("condition")["bp_controlled_flag"]
            .mean()
            .mul(100)
            .round(1)
            .sort_values(ascending=False)
        )
        st.bar_chart(bp_by_condition)


def render_multivariate_analysis() -> None:
    st.markdown("<h2 style='color:#0f766e;'>Multivariate Analysis</h2>", unsafe_allow_html=True)

    _, analysis_df = _load_reports()
    if analysis_df is None:
        st.info("Run `uv run eda_validation.py` first to generate the analysis dataset.")
        return

    st.markdown("<h3 style='color:#0f766e;'>Correlation Matrix</h3>", unsafe_allow_html=True)

    numeric_cols = [
        "age",
        "total_visits",
        "no_show_count",
        "ambulatory_visits",
        "pharmacy_visits",
        "no_show_rate",
        "hba1c_completed_flag",
        "bp_controlled_flag",
        "annual_checkup_flag",
        "diabetes_flag",
        "hypertension_flag",
    ]

    corr_df = analysis_df[numeric_cols].corr().round(2)

    fig, ax = plt.subplots(figsize=(11, 8))

    cmap = plt.cm.YlGnBu
    heatmap = ax.imshow(corr_df, cmap=cmap, vmin=-1, vmax=1)

    ax.set_xticks(range(len(corr_df.columns)))
    ax.set_yticks(range(len(corr_df.index)))
    ax.set_xticklabels(corr_df.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(corr_df.index, fontsize=9)

    for i in range(len(corr_df.index)):
        for j in range(len(corr_df.columns)):
            value = corr_df.iloc[i, j]
            text_color = "white" if abs(value) >= 0.5 else "black"
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8
            )

    cbar = plt.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Correlation", rotation=270, labelpad=15)

    ax.set_title("Feature Correlation Matrix", fontsize=12, color="#0f766e", pad=12)
    plt.tight_layout()

    st.pyplot(fig)

    st.markdown(
        "<h3 style='color:#0f766e;'>Average Visits by Condition and Annual Checkup Status</h3>",
        unsafe_allow_html=True
    )
    condition_checkup = (
        analysis_df.groupby(["condition", "annual_checkup"])["total_visits"]
        .mean()
        .reset_index()
    )
    pivot_condition_checkup = condition_checkup.pivot(
        index="condition",
        columns="annual_checkup",
        values="total_visits"
    ).fillna(0).round(2)
    st.dataframe(pivot_condition_checkup, use_container_width=True)

    st.markdown(
        "<h3 style='color:#0f766e;'>No-Show Rate by Condition and Dominant Department</h3>",
        unsafe_allow_html=True
    )
    dept_mix_df = analysis_df.copy()
    dept_mix_df["dominant_department"] = dept_mix_df.apply(
        lambda row: "ambulatory"
        if row["ambulatory_visits"] >= row["pharmacy_visits"]
        else "pharmacy",
        axis=1
    )

    no_show_mix = (
        dept_mix_df.groupby(["condition", "dominant_department"])["no_show_rate"]
        .mean()
        .reset_index()
    )

    pivot_no_show_mix = no_show_mix.pivot(
        index="condition",
        columns="dominant_department",
        values="no_show_rate"
    ).fillna(0).round(3)

    st.dataframe(pivot_no_show_mix, use_container_width=True)

    st.markdown("<h3 style='color:#0f766e;'>Sample Analytical Dataset</h3>", unsafe_allow_html=True)
    sample_cols = [
        "patient_id",
        "age",
        "condition",
        "total_visits",
        "no_show_rate",
        "annual_checkup",
        "hba1c_test_done",
        "bp_controlled",
    ]
    st.dataframe(analysis_df[sample_cols].head(20), use_container_width=True)