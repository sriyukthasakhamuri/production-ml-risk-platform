from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.services.prediction_logger import log_prediction


MODEL_PATH = Path("app/models/artifacts/best_model.joblib")

app = FastAPI(
    title="Production ML Risk Prediction API",
    description="Predict customer churn risk using the trained production ML model.",
    version="1.0.0",
)


if not MODEL_PATH.exists():
    raise RuntimeError(
        f"Model not found at {MODEL_PATH}. "
        "Run the model training pipeline first."
    )


model = joblib.load(MODEL_PATH)


class CustomerInput(BaseModel):
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


class PredictionResponse(BaseModel):
    churn_prediction: int
    churn_probability: float
    risk_level: str


def get_risk_level(probability: float) -> str:
    if probability >= 0.70:
        return "high"

    if probability >= 0.40:
        return "medium"

    return "low"


@app.get("/")
def root():
    return {
        "message": "Production ML Risk Prediction API",
        "status": "running",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(customer: CustomerInput):
    try:
        customer_data = customer.model_dump()

        tenure = customer_data["tenure"]
        total_charges = customer_data["TotalCharges"]

        customer_data["AvgMonthlySpend"] = (
            total_charges / tenure
            if tenure > 0
            else total_charges
        )

        input_df = pd.DataFrame(
            [customer_data]
        )

        prediction = int(
            model.predict(input_df)[0]
        )

        probability = float(
            model.predict_proba(input_df)[0][1]
        )

        risk_level = get_risk_level(
            probability
        )
        log_prediction(
           input_data=customer_data,
           churn_prediction=prediction,
           churn_probability=probability,
           risk_level=risk_level,
        )
 
        return PredictionResponse(
            churn_prediction=prediction,
            churn_probability=round(
                probability,
                4,
            ),
            risk_level=risk_level,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        ) from exc