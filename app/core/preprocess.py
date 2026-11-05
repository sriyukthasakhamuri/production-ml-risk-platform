from pathlib import Path

import pandas as pd

from app.core.data_loader import load_data


OUTPUT_PATH = Path("data/processed/telco_customer_churn_clean.csv")


def preprocess_data() -> pd.DataFrame:
    df = load_data().copy()

    # Convert TotalCharges to numeric.
    # Blank values become NaN.
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce",
    )

    # Remove rows with missing TotalCharges.
    df = df.dropna(subset=["TotalCharges"]).copy()

    # Convert target to numeric.
    df["Churn"] = df["Churn"].map(
        {
            "No": 0,
            "Yes": 1,
        }
    )

    # Remove customer identifier because it should not be used
    # as a predictive feature.
    df = df.drop(columns=["customerID"])

    # Create a useful engineered feature.
    df["AvgMonthlySpend"] = (
        df["TotalCharges"] / df["tenure"].replace(0, 1)
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    return df


if __name__ == "__main__":
    processed_data = preprocess_data()

    print("Preprocessing completed successfully.")
    print(f"Rows: {len(processed_data)}")
    print(f"Columns: {len(processed_data.columns)}")

    print("\nMissing values:")
    print(processed_data.isnull().sum().sum())

    print("\nChurn distribution:")
    print(processed_data["Churn"].value_counts())

    print(
        f"\nSaved processed dataset to: {OUTPUT_PATH}"
    )