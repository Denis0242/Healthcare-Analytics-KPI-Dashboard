import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

CURATED_PATH = "data/curated"
REPORTS_PATH = "reports"

np.random.seed(42)


def ensure_directories() -> None:
    os.makedirs(CURATED_PATH, exist_ok=True)
    os.makedirs(REPORTS_PATH, exist_ok=True)


def generate_patients(n: int = 550) -> pd.DataFrame:
    patient_ids = range(1, n + 1)
    ages = np.random.randint(18, 90, n)

    conditions = np.random.choice(
        ["diabetes", "hypertension", "none"],
        size=n,
        p=[0.30, 0.40, 0.30]
    )

    patients = pd.DataFrame({
        "patient_id": patient_ids,
        "age": ages,
        "condition": conditions
    })

    return patients


def generate_visits(patients: pd.DataFrame) -> pd.DataFrame:
    visits = []
    visit_id = 1

    for patient_id in patients["patient_id"]:
        num_visits = np.random.randint(1, 6)

        for _ in range(num_visits):
            visit_date = datetime.today() - timedelta(days=int(np.random.randint(0, 365)))
            department = np.random.choice(["ambulatory", "pharmacy"], p=[0.75, 0.25])
            no_show_flag = np.random.choice([0, 1], p=[0.90, 0.10])

            visits.append({
                "visit_id": visit_id,
                "patient_id": patient_id,
                "visit_date": visit_date.date(),
                "department": department,
                "no_show_flag": no_show_flag
            })
            visit_id += 1

    return pd.DataFrame(visits)


def generate_measures(patients: pd.DataFrame, visits: pd.DataFrame) -> pd.DataFrame:
    visit_counts = visits.groupby("patient_id").size().rename("visit_count").reset_index()
    annual_checkup_ids = set(visit_counts.loc[visit_counts["visit_count"] >= 1, "patient_id"])

    measures = []

    for _, row in patients.iterrows():
        patient_id = row["patient_id"]
        condition = row["condition"]

        hba1c_test_done = "yes"
        bp_controlled = "yes"
        annual_checkup = "yes"

        if condition == "diabetes":
            hba1c_test_done = "yes" if np.random.rand() < 0.82 else "no"
        else:
            hba1c_test_done = "no"

        if condition == "hypertension":
            bp_controlled = "yes" if np.random.rand() < 0.72 else "no"
        else:
            bp_controlled = "no"

        if patient_id in annual_checkup_ids:
            annual_checkup = "yes" if np.random.rand() < 0.96 else "no"
        else:
            annual_checkup = "no"

        measures.append({
            "patient_id": patient_id,
            "hba1c_test_done": hba1c_test_done,
            "bp_controlled": bp_controlled,
            "annual_checkup": annual_checkup
        })

    return pd.DataFrame(measures)


def save_data(patients: pd.DataFrame, visits: pd.DataFrame, measures: pd.DataFrame) -> None:
    patients.to_csv(os.path.join(CURATED_PATH, "patients.csv"), index=False)
    visits.to_csv(os.path.join(CURATED_PATH, "visits.csv"), index=False)
    measures.to_csv(os.path.join(CURATED_PATH, "measures.csv"), index=False)


def run_etl() -> None:
    print("Running ETL...")
    ensure_directories()

    patients = generate_patients(550)
    visits = generate_visits(patients)
    measures = generate_measures(patients, visits)

    save_data(patients, visits, measures)

    print("ETL completed successfully.")
    print(f"Saved: {os.path.join(CURATED_PATH, 'patients.csv')}")
    print(f"Saved: {os.path.join(CURATED_PATH, 'visits.csv')}")
    print(f"Saved: {os.path.join(CURATED_PATH, 'measures.csv')}")
    print(f"Patients rows: {len(patients)}")
    print(f"Visits rows: {len(visits)}")
    print(f"Measures rows: {len(measures)}")


if __name__ == "__main__":
    run_etl()