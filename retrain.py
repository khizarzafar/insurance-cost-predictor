"""
retrain.py — Regenerate model.pkl and scaler.pkl
Run this from your project folder:
    python retrain.py
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib
import sklearn

print(f"Using scikit-learn version: {sklearn.__version__}")

# ── Load data ─────────────────────────────────────────────────
df = pd.read_csv("insurance.csv")
df.drop_duplicates(inplace=True)

# ── Encode features ───────────────────────────────────────────
df["sex"]    = df["sex"].map({"male": 0, "female": 1})
df["smoker"] = df["smoker"].map({"no": 0, "yes": 1})
df.rename(columns={"sex": "is_female", "smoker": "is_smoker"}, inplace=True)
df = pd.get_dummies(df, columns=["region"], drop_first=True)
df = df.astype(int)

# ── BMI category ──────────────────────────────────────────────
df["bmi_category"] = pd.cut(
    df["bmi"],
    bins=[0, 18.5, 24.9, 29.9, float("inf")],
    labels=["underweight", "normal", "overweight", "obese"]
)
df = pd.get_dummies(df, columns=["bmi_category"], drop_first=True)
df = df.astype(int)

# ── Final features ────────────────────────────────────────────
FEATURES = ["age", "is_female", "bmi", "children",
            "is_smoker", "region_southeast", "bmi_category_obese"]
TARGET   = "charges"

X = df[FEATURES]
y = df[TARGET]

# ── Split FIRST, then scale ───────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
COLS_TO_SCALE = ["age", "bmi", "children"]
X_train[COLS_TO_SCALE] = scaler.fit_transform(X_train[COLS_TO_SCALE])
X_test[COLS_TO_SCALE]  = scaler.transform(X_test[COLS_TO_SCALE])

# ── Train ─────────────────────────────────────────────────────
model = LinearRegression()
model.fit(X_train, y_train)
r2 = r2_score(y_test, model.predict(X_test))
print(f"R² Score: {r2:.4f}")

# ── Save ──────────────────────────────────────────────────────
joblib.dump(model,  "model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("model.pkl saved!")
print("scaler.pkl saved!")
print("Done — ready for deployment!")
