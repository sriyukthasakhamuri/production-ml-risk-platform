from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


LOG_PATH = Path("reports/predictions.csv")


def log_prediction(
    input_data: dict,
    churn_prediction: int,
    churn_probability: float,
    risk_level: str,
) -> None:
    LOG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **input_data,
        "churn_prediction": churn_prediction,
        "churn_probability": churn_probability,
        "risk_level": risk_level,
    }

    new_row = pd.DataFrame([row])

    if LOG_PATH.exists():
        new_row.to_csv(
            LOG_PATH,
            mode="a",
            header=False,
            index=False,
        )
    else:
        new_row.to_csv(
            LOG_PATH,
            index=False,
        )