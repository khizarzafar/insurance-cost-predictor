# 🏥 Insurance Cost Predictor

A full-stack ML web application that predicts individual medical insurance charges based on personal attributes. Built with FastAPI, Streamlit, and Docker — deployed on Hugging Face Spaces.

---

## 🚀 Live Demo

👉 [Try it on Hugging Face Spaces](#) *(link will be updated after deployment)*

---

## 🧠 What This Project Does

Given a person's details, the app predicts how much they are likely to be charged for medical insurance.

**Input Features:**

| Feature | Description |
|---|---|
| Age | Age of the insured person |
| BMI | Body Mass Index |
| Children | Number of dependents |
| Smoker | Whether the person smokes (yes / no) |
| Region | Residential region in the US |

**Output:** Predicted insurance charges in USD 💵

---

## 🏗️ Project Structure

```
insurance-cost-predictor/
├── app/
│   ├── __init__.py
│   └── main.py              ← FastAPI backend with /predict endpoint
├── streamlit_app.py         ← Streamlit frontend UI
├── model.pkl                ← Trained Linear Regression model
├── scaler.pkl               ← StandardScaler for feature normalization
├── training_model.ipynb     ← Full model training notebook (EDA + ML)
├── retrain.py               ← Script to retrain the model from scratch
├── insurance.csv            ← Dataset (1338 records)
├── requirements.txt         ← All Python dependencies
├── Dockerfile               ← Container build instructions
├── docker-compose.yml       ← Runs FastAPI + Streamlit together
├── start.sh                 ← Startup script for containers
└── README.md
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| ML Model | Scikit-learn (Linear Regression) |
| Backend API | FastAPI + Uvicorn |
| Frontend UI | Streamlit |
| Containerization | Docker + Docker Compose |
| Deployment | Hugging Face Spaces |

---

## 🐳 Run Locally with Docker

Make sure **Docker Desktop is running**, then:

```bash
# Clone the repo
git clone https://github.com/khizarzafar/insurance-cost-predictor.git
cd insurance-cost-predictor

# Build and run everything with one command
docker-compose up --build
```

Then open your browser:

- 🌐 **Streamlit UI** → http://localhost:8501
- 📄 **FastAPI Docs** → http://localhost:8000/docs

To stop all containers:

```bash
docker-compose down
```

---

## 🧪 Run Without Docker

```bash
# Install dependencies
pip install -r requirements.txt

# Terminal 1 — Start FastAPI backend
uvicorn app.main:app --reload --port 8000

# Terminal 2 — Start Streamlit frontend
streamlit run streamlit_app.py
```

---

## 📡 API Reference

**Endpoint:** `POST /predict`

**Request body (JSON):**
```json
{
  "age": 30,
  "bmi": 27.5,
  "children": 2,
  "smoker": "yes",
  "region": "southeast"
}
```

**Response:**
```json
{
  "predicted_charges": 18452.73
}
```

You can test it interactively at `http://localhost:8000/docs` using the built-in Swagger UI.

---

## 📊 Model Details

- **Algorithm:** Linear Regression
- **Training data:** `insurance.csv` (1,338 records)
- **Target variable:** `charges` (medical insurance cost in USD)
- **Key predictors:** Smoker status (strongest), Age, BMI
- **Preprocessing:** StandardScaler for numeric features, One-hot encoding for categoricals
- **Evaluation metrics:** R² Score and Adjusted R²

---

## 📓 Training Notebook

The `training_model.ipynb` notebook covers the full pipeline:

1. **EDA** — distributions, correlations, visualizations
2. **Data Cleaning** — duplicates, encoding, type fixes
3. **Feature Engineering** — BMI categories, one-hot encoding
4. **Feature Selection** — Pearson Correlation + Chi-Square tests
5. **Model Training** — 80/20 split, Linear Regression
6. **Evaluation** — R² and Adjusted R² on test set

---

## 📦 Dataset

**File:** `insurance.csv`

| Column | Type | Description |
|---|---|---|
| `age` | int | Age of insured person |
| `sex` | string | Gender (male / female) |
| `bmi` | float | Body Mass Index |
| `children` | int | Number of dependents |
| `smoker` | string | Smoking status (yes / no) |
| `region` | string | US region (northeast / northwest / southeast / southwest) |
| `charges` | float | Medical insurance charges — **target variable** |

---

## 👤 Author

**Khizar Zafar**
BSAI Student — Air University, Islamabad
[GitHub](https://github.com/khizarzafar)