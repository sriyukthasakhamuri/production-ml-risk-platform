from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path("data/processed/telco_customer_churn_clean.csv")
MODEL_DIR = Path("app/models/artifacts")


def load_processed_data():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    return X, y


def build_preprocessor(X, dense_output=False):
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_features = X.select_dtypes(exclude=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                numeric_features,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=not dense_output,
                ),
                categorical_features,
            ),
        ]
    )

    return preprocessor


def evaluate_model(name, model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }

    print(f"\n{name}")
    print("-" * len(name))

    for metric_name, value in metrics.items():
        if metric_name != "model":
            print(f"{metric_name}: {value:.4f}")

    print("\nClassification report:")
    print(classification_report(y_test, predictions))

    return metrics


def train_models():
    X, y = load_processed_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model_configs = [
        (
            "logistic_regression",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
            False,
        ),
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_split=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
            False,
        ),
        (
            "hist_gradient_boosting",
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=250,
                max_leaf_nodes=20,
                l2_regularization=1.0,
                random_state=42,
            ),
            True,
        ),
    ]

    results = []
    trained_models = {}

    for name, classifier, dense_output in model_configs:
        print(f"\nTraining {name}...")

        preprocessor = build_preprocessor(
            X_train,
            dense_output=dense_output,
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier),
            ]
        )

        pipeline.fit(X_train, y_train)

        metrics = evaluate_model(
            name,
            pipeline,
            X_test,
            y_test,
        )

        results.append(metrics)
        trained_models[name] = pipeline

    results_df = pd.DataFrame(results)

    best_row = results_df.sort_values(
        by="roc_auc",
        ascending=False,
    ).iloc[0]

    best_model_name = best_row["model"]
    best_model = trained_models[best_model_name]

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = MODEL_DIR / "best_model.joblib"

    joblib.dump(
        best_model,
        model_path,
    )

    results_path = MODEL_DIR / "model_comparison.csv"

    results_df.to_csv(
        results_path,
        index=False,
    )

    print("\n==============================")
    print("MODEL COMPARISON")
    print("==============================")

    print(
        results_df[
            [
                "model",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
            ]
        ].to_string(index=False)
    )

    print("\nBest model:")
    print(best_model_name)

    print("\nBest model ROC-AUC:")
    print(f"{best_row['roc_auc']:.4f}")

    print("\nSaved model to:")
    print(model_path)

    print("\nSaved model comparison to:")
    print(results_path)


if __name__ == "__main__":
    train_models()