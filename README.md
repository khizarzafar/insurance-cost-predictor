# insurance-cost-predictor
A machine learning model to predict individual medical insurance charges based on personal attributes such as age, sex, BMI, region, and number of dependents.
# 🏥 Insurance Cost Predictor

A machine learning project to predict individual medical insurance charges based on personal attributes such as age, sex, BMI, region, and number of dependents.

---

## 📌 Problem Statement

Medical insurance costs vary significantly across individuals. This project builds a predictive model that estimates annual insurance charges for a person based on demographic and health-related features. Accurate cost prediction can assist insurance companies, healthcare planners, and individuals in making informed financial decisions.

---

## 📊 Dataset

**Source:** [Medical Cost Personal Dataset — Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance)

| Feature    | Type        | Description                              |
|------------|-------------|------------------------------------------|
| `age`      | Numerical   | Age of the primary beneficiary           |
| `sex`      | Categorical | Gender of the policyholder (male/female) |
| `bmi`      | Numerical   | Body Mass Index                          |
| `children` | Numerical   | Number of dependents covered             |
| `smoker`   | Categorical | Whether the policyholder smokes          |
| `region`   | Categorical | Residential region in the US             |
| `charges`  | Numerical   | 🎯 Target — Annual medical insurance cost |

---

## 🎯 Objective

- Perform Exploratory Data Analysis (EDA) to uncover patterns
- Preprocess and encode features for model training
- Train and evaluate multiple regression models
- Identify the best-performing model for deployment

---

## 🧪 Models Used

- Linear Regression
- Ridge / Lasso Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting (XGBoost / LightGBM)

---

## 📁 Project Structure

```
insurance-cost-predictor/
│
├── data/
│   ├── raw/                  # Original dataset (do not modify)
│   └── processed/            # Cleaned & encoded data
│
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
│
├── src/
│   ├── preprocess.py         # Data cleaning & feature engineering
│   ├── train.py              # Model training pipeline
│   └── evaluate.py           # Metrics & evaluation
│
├── models/
│   └── best_model.pkl        # Saved trained model
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/insurance-cost-predictor.git
cd insurance-cost-predictor
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 📈 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| MAE    | Mean Absolute Error |
| RMSE   | Root Mean Squared Error |
| R²     | Coefficient of Determination |

---

## 🔍 Key Findings *(to be updated after EDA)*

- [ ] Distribution of insurance charges
- [ ] Impact of smoking on charges
- [ ] Correlation between BMI and charges
- [ ] Regional cost variations

---

## 🚀 Future Improvements

- Deploy model as a REST API using Flask or FastAPI
- Build a simple frontend for user input and cost prediction
- Experiment with Neural Networks for improved accuracy

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 👤 Author

**Khizar Hayat Zafar**
- GitHub: https://github.com/khizarzafar
- LinkedIn: www.linkedin.com/in/khizar-hayyat-zafar-b46907286
