from pathlib import Path

import joblib
import pandas as pd

from app.core.preprocess import preprocess_data


MODEL_PATH = Path(
    "app/models/artifacts/best_model.joblib"
)

PROCESSED_DATA_PATH = Path(
    "data/processed/telco_customer_churn_clean.csv"
)


def test_preprocessing_pipeline():
    df = preprocess_data()

    assert not df.empty

    assert "Churn" in df.columns
    assert "AvgMonthlySpend" in df.columns
    assert "customerID" not in df.columns

    assert df.isnull().sum().sum() == 0

    assert PROCESSED_DATA_PATH.exists()


def test_model_artifact_exists():
    assert MODEL_PATH.exists()


def test_model_can_be_loaded():
    model = joblib.load(
        MODEL_PATH
    )

    assert model is not None


def test_processed_target_values():
    df = pd.read_csv(
        PROCESSED_DATA_PATH
    )

    target_values = set(
        df["Churn"].unique()
    )

    assert target_values.issubset(
        {0, 1}
    )