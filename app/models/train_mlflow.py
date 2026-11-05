from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
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
MLFLOW_ARTIFACT_DIR = Path("reports/mlflow_models")

EXPERIMENT_NAME = "telco-churn-model-comparison"


def load_processed_data():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    return X, y


def build_preprocessor(X, dense_output=False):
    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numeric_features = X.select_dtypes(
        exclude=["object"]
    ).columns.tolist()

    return ColumnTransformer(
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


def calculate_metrics(model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(
            y_test,
            predictions,
        ),
        "precision": precision_score(
            y_test,
            predictions,
        ),
        "recall": recall_score(
            y_test,
            predictions,
        ),
        "f1": f1_score(
            y_test,
            predictions,
        ),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities,
        ),
    }


def train_with_mlflow():
    X, y = load_processed_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    mlflow.set_experiment(EXPERIMENT_NAME)

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MLFLOW_ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
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
                (
                    "preprocessor",
                    preprocessor,
                ),
                (
                    "classifier",
                    classifier,
                ),
            ]
        )

        with mlflow.start_run(run_name=name):

            pipeline.fit(
                X_train,
                y_train,
            )

            metrics = calculate_metrics(
                pipeline,
                X_test,
                y_test,
            )

            mlflow.log_param(
                "model_name",
                name,
            )

            mlflow.log_param(
                "test_size",
                0.20,
            )

            mlflow.log_param(
                "random_state",
                42,
            )

            classifier_params = classifier.get_params()

            for key, value in classifier_params.items():

                if isinstance(
                    value,
                    (
                        str,
                        int,
                        float,
                        bool,
                    ),
                ) or value is None:

                    mlflow.log_param(
                        key,
                        value,
                    )

            for metric_name, metric_value in metrics.items():

                mlflow.log_metric(
                    metric_name,
                    metric_value,
                )

            artifact_path = (
                MLFLOW_ARTIFACT_DIR
                / f"{name}.joblib"
            )

            joblib.dump(
                pipeline,
                artifact_path,
            )

            mlflow.log_artifact(
                str(artifact_path),
                artifact_path="model",
            )

            results.append(
                {
                    "model": name,
                    **metrics,
                }
            )

            trained_models[name] = pipeline

            print(f"\n{name}")

            for metric_name, value in metrics.items():

                print(
                    f"{metric_name}: "
                    f"{value:.4f}"
                )

    results_df = pd.DataFrame(
        results
    )

    best_row = results_df.sort_values(
        by="roc_auc",
        ascending=False,
    ).iloc[0]

    best_model_name = best_row["model"]

    best_model = trained_models[
        best_model_name
    ]

    best_model_path = (
        MODEL_DIR
        / "best_model.joblib"
    )

    joblib.dump(
        best_model,
        best_model_path,
    )

    results_path = (
        MODEL_DIR
        / "mlflow_model_comparison.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    print("\n==============================")
    print("MLFLOW MODEL COMPARISON")
    print("==============================")

    print(
        results_df.to_string(
            index=False
        )
    )

    print("\nBest model:")
    print(best_model_name)

    print("\nBest ROC-AUC:")
    print(
        f"{best_row['roc_auc']:.4f}"
    )

    print("\nSaved best model:")
    print(best_model_path)

    print("\nSaved comparison:")
    print(results_path)


if __name__ == "__main__":
    train_with_mlflow()