from pathlib import Path

import numpy as np
import pandas as pd


REFERENCE_PATH = Path(
    "data/processed/telco_customer_churn_clean.csv"
)

OUTPUT_PATH = Path(
    "reports/synthetic_production_batch.csv"
)

RANDOM_SEED = 42
SAMPLE_SIZE = 300


def generate_production_batch() -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)

    reference = pd.read_csv(
        REFERENCE_PATH
    )

    sample = reference.sample(
        n=SAMPLE_SIZE,
        replace=True,
        random_state=RANDOM_SEED,
    ).copy()

    # Introduce controlled production drift.

    sample["MonthlyCharges"] = (
        sample["MonthlyCharges"] * 1.18
    )

    sample["TotalCharges"] = (
        sample["TotalCharges"] * 1.10
    )

    sample["tenure"] = (
        sample["tenure"] * 0.65
    ).round().astype(int)

    sample["tenure"] = sample[
        "tenure"
    ].clip(lower=0)

    sample["Contract"] = rng.choice(
        [
            "Month-to-month",
            "One year",
            "Two year",
        ],
        size=len(sample),
        p=[0.75, 0.18, 0.07],
    )

    sample["InternetService"] = rng.choice(
        [
            "Fiber optic",
            "DSL",
            "No",
        ],
        size=len(sample),
        p=[0.70, 0.25, 0.05],
    )

    sample["PaymentMethod"] = rng.choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        size=len(sample),
        p=[0.60, 0.15, 0.15, 0.10],
    )

    sample["AvgMonthlySpend"] = (
        sample["TotalCharges"]
        / sample["tenure"].replace(0, 1)
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sample.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    return sample


if __name__ == "__main__":
    batch = generate_production_batch()

    print(
        "Synthetic production batch created successfully."
    )

    print(
        f"Rows generated: {len(batch)}"
    )

    print(
        f"Saved to: {OUTPUT_PATH}"
    )

    print(
        "\nAverage MonthlyCharges:"
    )

    print(
        round(
            batch["MonthlyCharges"].mean(),
            2,
        )
    )

    print(
        "\nAverage tenure:"
    )

    print(
        round(
            batch["tenure"].mean(),
            2,
        )
    )

    print(
        "\nContract distribution:"
    )

    print(
        batch["Contract"].value_counts(
            normalize=True
        )
    )