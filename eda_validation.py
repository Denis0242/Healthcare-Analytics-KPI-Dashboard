import os
import pandas as pd

CURATED_PATH = "data/curated"
REPORTS_PATH = "reports"


def ensure_directories() -> None:
    os.makedirs(CURATED_PATH, exist_ok=True)
    os.makedirs(REPORTS_PATH, exist_ok=True)


def load_data():
    patients = pd.read_csv(os.path.join(CURATED_PATH, "patients.csv"))
    visits = pd.read_csv(os.path.join(CURATED_PATH, "visits.csv"), parse_dates=["visit_date"])
    measures = pd.read_csv(os.path.join(CURATED_PATH, "measures.csv"))
    return patients, visits, measures


def check_duplicates(df: pd.DataFrame, table_name: str) -> int:
    duplicate_count = int(df.duplicated().sum())
    print(f"[{table_name}] Duplicate rows: {duplicate_count}")
    return duplicate_count


def check_missing_values(df: pd.DataFrame, table_name: str) -> pd.Series:
    missing_values = df.isnull().sum()
    print(f"\n[{table_name}] Missing values:")
    print(missing_values)
    return missing_values


def validate_counts(patients: pd.DataFrame, visits: pd.DataFrame, measures: pd.DataFrame) -> dict:
    print("\n[COUNT VALIDATION]")
    print(f"Patients row count: {len(patients)}")
    print(f"Visits row count: {len(visits)}")
    print(f"Measures row count: {len(measures)}")

    unique_patient_ids_patients = patients["patient_id"].nunique()
    unique_patient_ids_visits = visits["patient_id"].nunique()
    unique_patient_ids_measures = measures["patient_id"].nunique()

    print(f"Unique patient_id in patients: {unique_patient_ids_patients}")
    print(f"Unique patient_id in visits: {unique_patient_ids_visits}")
    print(f"Unique patient_id in measures: {unique_patient_ids_measures}")

    patient_measure_status = (
        "PASS" if unique_patient_ids_patients == unique_patient_ids_measures else "WARNING"
    )
    visit_patient_status = (
        "PASS" if unique_patient_ids_visits <= unique_patient_ids_patients else "WARNING"
    )

    print(f"{patient_measure_status}: patients and measures patient counts align")
    print(f"{visit_patient_status}: visits patient_ids are within patients table")

    return {
        "patients_row_count": len(patients),
        "visits_row_count": len(visits),
        "measures_row_count": len(measures),
        "unique_patient_ids_patients": unique_patient_ids_patients,
        "unique_patient_ids_visits": unique_patient_ids_visits,
        "unique_patient_ids_measures": unique_patient_ids_measures,
        "patient_measure_status": patient_measure_status,
        "visit_patient_status": visit_patient_status,
    }


def save_validation_summary(patients: pd.DataFrame, visits: pd.DataFrame, measures: pd.DataFrame) -> str:
    summary_rows = []

    for table_name, df in {
        "patients": patients,
        "visits": visits,
        "measures": measures
    }.items():
        summary_rows.append({
            "table_name": table_name,
            "row_count": len(df),
            "column_count": df.shape[1],
            "duplicate_rows": int(df.duplicated().sum()),
            "missing_values": int(df.isnull().sum().sum())
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_file = os.path.join(REPORTS_PATH, "validation_summary.csv")
    summary_df.to_csv(summary_file, index=False)

    print("\n=== VALIDATION SUMMARY SAVED ===")
    print(f"File: {summary_file}")
    print(summary_df)

    return summary_file


def build_analysis_dataset(
    patients: pd.DataFrame,
    visits: pd.DataFrame,
    measures: pd.DataFrame
) -> pd.DataFrame:
    visit_summary = (
        visits.groupby("patient_id")
        .agg(
            total_visits=("visit_id", "count"),
            no_show_count=("no_show_flag", "sum"),
            ambulatory_visits=("department", lambda s: (s == "ambulatory").sum()),
            pharmacy_visits=("department", lambda s: (s == "pharmacy").sum()),
            latest_visit_date=("visit_date", "max")
        )
        .reset_index()
    )

    analysis_df = (
        patients.merge(measures, on="patient_id", how="left")
        .merge(visit_summary, on="patient_id", how="left")
    )

    fill_cols = ["total_visits", "no_show_count", "ambulatory_visits", "pharmacy_visits"]
    analysis_df[fill_cols] = analysis_df[fill_cols].fillna(0)

    analysis_df["no_show_rate"] = analysis_df.apply(
        lambda row: row["no_show_count"] / row["total_visits"] if row["total_visits"] > 0 else 0,
        axis=1
    )

    analysis_df["hba1c_completed_flag"] = (analysis_df["hba1c_test_done"] == "yes").astype(int)
    analysis_df["bp_controlled_flag"] = (analysis_df["bp_controlled"] == "yes").astype(int)
    analysis_df["annual_checkup_flag"] = (analysis_df["annual_checkup"] == "yes").astype(int)
    analysis_df["diabetes_flag"] = (analysis_df["condition"] == "diabetes").astype(int)
    analysis_df["hypertension_flag"] = (analysis_df["condition"] == "hypertension").astype(int)

    analysis_file = os.path.join(REPORTS_PATH, "analysis_dataset.csv")
    analysis_df.to_csv(analysis_file, index=False)

    print("\n=== ANALYSIS DATASET SAVED ===")
    print(f"File: {analysis_file}")

    return analysis_df


def main() -> None:
    ensure_directories()
    print("=== DATA VALIDATION START ===\n")

    patients, visits, measures = load_data()

    check_duplicates(patients, "patients")
    check_duplicates(visits, "visits")
    check_duplicates(measures, "measures")

    check_missing_values(patients, "patients")
    check_missing_values(visits, "visits")
    check_missing_values(measures, "measures")

    validate_counts(patients, visits, measures)
    summary_file = save_validation_summary(patients, visits, measures)
    analysis_df = build_analysis_dataset(patients, visits, measures)

    print("\n=== OUTPUT CHECK ===")
    print(f"Validation summary exists: {os.path.exists(summary_file)}")
    print(f"Analysis dataset rows: {len(analysis_df)}")
    print("\n=== DATA VALIDATION COMPLETE ===")


if __name__ == "__main__":
    main()