from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/raw/telco_customer_churn.csv")


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Make sure telco_customer_churn.csv exists in data/raw/."
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = {
        "customerID",
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
        "Churn",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_columns)}"
        )

    if df.empty:
        raise ValueError("Dataset is empty.")

    return df


if __name__ == "__main__":
    data = load_data()

    print("Dataset loaded successfully.")
    print(f"Rows: {len(data)}")
    print(f"Columns: {len(data.columns)}")
    print("\nColumn names:")
    print(list(data.columns))

    print("\nChurn distribution:")
    print(data["Churn"].value_counts(dropna=False))