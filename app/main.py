"""
Insurance Charges Predictor — FastAPI Backend
==============================================
This file is the heart of the API. It:
  1. Loads the trained model and scaler from disk (once, at startup)
  2. Exposes a POST /predict endpoint that takes patient data and returns a charge estimate
  3. Exposes a GET /health endpoint so deployment platforms can check if the API is alive

Run locally with:
    uvicorn app.main:app --reload
"""

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

# ─────────────────────────────────────────────
# 1.  App Initialization
# ─────────────────────────────────────────────

app = FastAPI(
    title="Insurance Charges Predictor",
    description="Predicts annual insurance charges based on patient demographics.",
    version="1.0.0",
)

# ─────────────────────────────────────────────
# 2.  Load Model & Scaler at Startup
# ─────────────────────────────────────────────
# We load these ONCE when the server starts, not on every request.
# Loading on every request would be extremely slow.

try:
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    print("✅ model.pkl and scaler.pkl loaded successfully.")
except FileNotFoundError as e:
    # If the files are missing, crash loudly — better than silent wrong predictions
    raise RuntimeError(
        "Could not find model.pkl or scaler.pkl. "
        "Run the training notebook first to generate these files."
    ) from e


# ─────────────────────────────────────────────
# 3.  Input Schema (Pydantic Model)
# ─────────────────────────────────────────────
# Pydantic validates ALL incoming data automatically.
# If a field is missing or the wrong type, FastAPI returns a 422 error
# with a clear message — we don't need to write that validation ourselves.

class PatientInput(BaseModel):
    """The data a user must send to get a prediction."""

    age: int = Field(
        ...,                   # '...' means this field is required
        ge=18,                 # ge = greater-than-or-equal
        le=100,
        description="Patient age in years (18–100)",
        example=35,
    )
    sex: str = Field(
        ...,
        description="Patient sex: 'male' or 'female'",
        example="male",
    )
    bmi: float = Field(
        ...,
        ge=10.0,
        le=60.0,
        description="Body Mass Index (10.0–60.0)",
        example=27.5,
    )
    children: int = Field(
        ...,
        ge=0,
        le=10,
        description="Number of children covered by insurance (0–10)",
        example=2,
    )
    smoker: str = Field(
        ...,
        description="Smoker status: 'yes' or 'no'",
        example="no",
    )
    region: str = Field(
        ...,
        description="US region: 'northeast', 'northwest', 'southeast', or 'southwest'",
        example="southeast",
    )

    # ── Validators ──────────────────────────────────────────────────────────
    # These run automatically when the object is created.
    # They normalise the string inputs so 'Male', 'MALE', 'male' all work.

    @field_validator("sex")
    @classmethod
    def validate_sex(cls, v):
        v = v.strip().lower()
        if v not in ("male", "female"):
            raise ValueError("sex must be 'male' or 'female'")
        return v

    @field_validator("smoker")
    @classmethod
    def validate_smoker(cls, v):
        v = v.strip().lower()
        if v not in ("yes", "no"):
            raise ValueError("smoker must be 'yes' or 'no'")
        return v

    @field_validator("region")
    @classmethod
    def validate_region(cls, v):
        v = v.strip().lower()
        valid = ("northeast", "northwest", "southeast", "southwest")
        if v not in valid:
            raise ValueError(f"region must be one of: {valid}")
        return v


# ─────────────────────────────────────────────
# 4.  Output Schema (Pydantic Model)
# ─────────────────────────────────────────────
# Defining the response shape makes the API self-documenting
# and prevents accidentally leaking extra fields.

class PredictionOutput(BaseModel):
    predicted_charges: float = Field(description="Estimated annual insurance charges in USD")
    features_used: dict = Field(description="The exact feature values sent to the model")


# ─────────────────────────────────────────────
# 5.  Preprocessing Helper
# ─────────────────────────────────────────────
# This function must mirror your notebook's preprocessing EXACTLY.
# If they differ, predictions will be wrong — even if the model loaded fine.

# These are the 7 columns the model was trained on, in the same order.
FEATURE_ORDER = [
    "age", "is_female", "bmi", "children",
    "is_smoker", "region_southeast", "bmi_category_obese",
]

# These are the 3 columns the scaler was fitted on (same order as in notebook).
COLS_TO_SCALE = ["age", "bmi", "children"]


def preprocess(data: PatientInput) -> pd.DataFrame:
    """
    Convert raw PatientInput into a single-row DataFrame that matches
    the exact format the model expects.
    """

    # ── Binary Encodings ────────────────────────────────────────────────────
    is_female          = 1 if data.sex == "female" else 0
    is_smoker          = 1 if data.smoker == "yes" else 0

    # ── Region: One-hot (only southeast is a model feature) ─────────────────
    region_southeast   = 1 if data.region == "southeast" else 0

    # ── BMI Category: Obese flag (bmi > 29.9) ───────────────────────────────
    bmi_category_obese = 1 if data.bmi > 29.9 else 0

    # ── Assemble into a single-row DataFrame ────────────────────────────────
    row = {
        "age":               data.age,
        "is_female":         is_female,
        "bmi":               data.bmi,
        "children":          data.children,
        "is_smoker":         is_smoker,
        "region_southeast":  region_southeast,
        "bmi_category_obese": bmi_category_obese,
    }

    df = pd.DataFrame([row], columns=FEATURE_ORDER)

    # ── Scale continuous columns ─────────────────────────────────────────────
    # We use .transform() (not fit_transform) — the scaler already knows
    # the mean and std from the training data. We just apply those.
    df[COLS_TO_SCALE] = scaler.transform(df[COLS_TO_SCALE])

    return df


# ─────────────────────────────────────────────
# 6.  API Endpoints
# ─────────────────────────────────────────────

@app.get("/health", tags=["Utility"])
def health_check():
    """
    Returns 200 OK if the API is running.
    Deployment platforms (Docker, Hugging Face) ping this route
    to know whether the container is healthy and ready to serve traffic.
    """
    return {"status": "healthy", "model": "Insurance LinearRegression v1.0"}


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
def predict(patient: PatientInput):
    """
    Accepts patient demographics and returns a predicted annual insurance charge.

    **Example request body:**
    ```json
    {
      "age": 35,
      "sex": "male",
      "bmi": 27.5,
      "children": 2,
      "smoker": "no",
      "region": "southeast"
    }
    ```
    """
    try:
        # Preprocess the input into model-ready format
        X = preprocess(patient)

        # Run inference
        prediction = model.predict(X)[0]

        # Clamp to a realistic minimum — model can occasionally predict negative
        # values for very low-risk patients; charges can't be negative
        prediction = max(prediction, 0.0)

        return PredictionOutput(
            predicted_charges=round(prediction, 2),
            features_used=X.to_dict(orient="records")[0],
        )

    except Exception as e:
        # If something unexpected goes wrong, return a clean error (not a crash)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
