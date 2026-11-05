from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset


REFERENCE_PATH = Path(
    "data/processed/telco_customer_churn_clean.csv"
)

CURRENT_PATH = Path(
    "reports/synthetic_production_batch.csv"
)

COMPARISON_PATH = Path(
    "app/models/artifacts/mlflow_model_comparison.csv"
)

RETRAINING_LOG_PATH = Path(
    "reports/retraining_events.csv"
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

DRIFT_THRESHOLD = 0.50


def load_data():
    reference = pd.read_csv(
        REFERENCE_PATH
    )[MONITORED_COLUMNS]

    current = pd.read_csv(
        CURRENT_PATH
    )[MONITORED_COLUMNS]

    return reference, current


def calculate_drift_share() -> float:
    reference, current = load_data()

    report = Report(
        metrics=[
            DataDriftPreset(),
        ]
    )

    result = report.run(
        reference_data=reference,
        current_data=current,
    )

    result_dict = result.dict()

    drift_share = None

    for metric in result_dict.get(
        "metrics",
        []
    ):
        value = metric.get("value")

        if isinstance(value, dict):
            if "share" in value:
                drift_share = value["share"]
                break

            if "share_of_drifted_columns" in value:
                drift_share = value[
                    "share_of_drifted_columns"
                ]
                break

    if drift_share is None:
        raise RuntimeError(
            "Could not extract drift share "
            "from Evidently report."
        )

    return float(drift_share)


def get_best_model_result():
    if not COMPARISON_PATH.exists():
        return None, None

    comparison = pd.read_csv(
        COMPARISON_PATH
    )

    if comparison.empty:
        return None, None

    best_row = comparison.sort_values(
        by="roc_auc",
        ascending=False,
    ).iloc[0]

    return (
        str(best_row["model"]),
        float(best_row["roc_auc"]),
    )


def log_retraining_event(
    drift_share: float,
    retraining_triggered: bool,
    best_model: str | None,
    best_roc_auc: float | None,
):
    RETRAINING_LOG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    row = {
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "drift_share": drift_share,
        "drift_threshold": DRIFT_THRESHOLD,
        "retraining_triggered": retraining_triggered,
        "best_model": best_model,
        "best_roc_auc": best_roc_auc,
    }

    new_row = pd.DataFrame(
        [row]
    )

    if RETRAINING_LOG_PATH.exists():
        new_row.to_csv(
            RETRAINING_LOG_PATH,
            mode="a",
            header=False,
            index=False,
        )
    else:
        new_row.to_csv(
            RETRAINING_LOG_PATH,
            index=False,
        )


def trigger_retraining():
    print(
        "\nStarting model retraining..."
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "app.models.train_mlflow",
        ],
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Model retraining failed."
        )

    print(
        "\nRetraining completed successfully."
    )


def main():
    drift_share = calculate_drift_share()

    print(
        f"Detected drift share: "
        f"{drift_share:.2%}"
    )

    print(
        f"Retraining threshold: "
        f"{DRIFT_THRESHOLD:.0%}"
    )

    retraining_triggered = False
    best_model = None
    best_roc_auc = None

    if drift_share >= DRIFT_THRESHOLD:
        print(
            "\nDrift threshold exceeded."
        )

        retraining_triggered = True

        trigger_retraining()

        best_model, best_roc_auc = (
            get_best_model_result()
        )

    else:
        print(
            "\nDrift below threshold. "
            "Retraining not required."
        )

        best_model, best_roc_auc = (
            get_best_model_result()
        )

    log_retraining_event(
        drift_share=drift_share,
        retraining_triggered=retraining_triggered,
        best_model=best_model,
        best_roc_auc=best_roc_auc,
    )

    print(
        "\nRetraining event logged to:"
    )

    print(
        RETRAINING_LOG_PATH
    )

    if best_model is not None:
        print(
            f"\nBest model: {best_model}"
        )

    if best_roc_auc is not None:
        print(
            f"Best ROC-AUC: "
            f"{best_roc_auc:.4f}"
        )


if __name__ == "__main__":
    main()