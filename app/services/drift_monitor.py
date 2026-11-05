from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset


REFERENCE_PATH = Path(
    "data/processed/telco_customer_churn_clean.csv"
)

CURRENT_PATH = Path(
    "reports/synthetic_production_batch.csv"
)

REPORT_PATH = Path(
    "reports/drift_report.html"
)


MONITORED_COLUMNS = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "AvgMonthlySpend",
    "Contract",
    "InternetService",
    "PaymentMethod",
]


def load_reference_data() -> pd.DataFrame:
    reference = pd.read_csv(
        REFERENCE_PATH
    )

    return reference[
        MONITORED_COLUMNS
    ].copy()


def load_current_data() -> pd.DataFrame:
    current = pd.read_csv(
        CURRENT_PATH
    )

    missing_columns = (
        set(MONITORED_COLUMNS)
        - set(current.columns)
    )

    if missing_columns:
        raise ValueError(
            "Production batch is missing "
            f"columns: {sorted(missing_columns)}"
        )

    return current[
        MONITORED_COLUMNS
    ].copy()


def generate_drift_report():
    if not CURRENT_PATH.exists():
        raise FileNotFoundError(
            "Synthetic production batch not found. "
            "Run generate_production_traffic first."
        )

    reference_data = load_reference_data()
    current_data = load_current_data()

    report = Report(
        metrics=[
            DataDriftPreset(),
        ]
    )

    result = report.run(
        reference_data=reference_data,
        current_data=current_data,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.save_html(
        str(REPORT_PATH)
    )

    print(
        "Drift report generated successfully."
    )

    print(
        f"Reference rows: "
        f"{len(reference_data)}"
    )

    print(
        f"Current production rows: "
        f"{len(current_data)}"
    )

    print(
        f"Saved report to: "
        f"{REPORT_PATH}"
    )


if __name__ == "__main__":
    generate_drift_report()