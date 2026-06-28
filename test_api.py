"""
Manual API Test Script
======================
Run the FastAPI server first:
    uvicorn app.main:app --reload

Then run this file in a separate terminal:
    python test_api.py

It sends 4 test cases and prints what the API returns.
"""

import requests

BASE_URL = "http://127.0.0.1:8000"

# ── Test Cases ────────────────────────────────────────────────────────────────
test_cases = [
    {
        "name": "Young non-smoker (expect LOW charges)",
        "data": {
            "age": 22, "sex": "female", "bmi": 21.0,
            "children": 0, "smoker": "no", "region": "northwest",
        },
    },
    {
        "name": "Obese smoker (expect HIGH charges)",
        "data": {
            "age": 55, "sex": "male", "bmi": 38.5,
            "children": 1, "smoker": "yes", "region": "southeast",
        },
    },
    {
        "name": "Middle-aged with children",
        "data": {
            "age": 40, "sex": "female", "bmi": 27.5,
            "children": 3, "smoker": "no", "region": "northeast",
        },
    },
    {
        "name": "Invalid input — should return 422 error",
        "data": {
            "age": 40, "sex": "alien",   # invalid sex value
            "bmi": 27.5, "children": 0,
            "smoker": "no", "region": "northeast",
        },
    },
]

# ── Health Check ─────────────────────────────────────────────────────────────
print("=" * 55)
print("  HEALTH CHECK")
print("=" * 55)
response = requests.get(f"{BASE_URL}/health")
print(f"  Status: {response.status_code}")
print(f"  Body:   {response.json()}")
print()

# ── Prediction Tests ──────────────────────────────────────────────────────────
print("=" * 55)
print("  PREDICTION TESTS")
print("=" * 55)

for case in test_cases:
    print(f"\n🔹 {case['name']}")
    response = requests.post(f"{BASE_URL}/predict", json=case["data"])

    if response.status_code == 200:
        result = response.json()
        print(f"   ✅ Predicted charges: ${result['predicted_charges']:,.2f}")
    else:
        print(f"   ❌ Error {response.status_code}: {response.json()['detail']}")
