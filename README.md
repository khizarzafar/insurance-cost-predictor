# Insurance Charges Prediction — EDA & Linear Regression

A machine learning project that analyzes an insurance dataset to understand what factors drive medical charges, and builds a regression model to predict them.

---

## What This Project Does

The notebook walks through the full pipeline — from raw data to a working prediction model:

- Exploratory data analysis to understand the data and spot patterns
- Data cleaning and preprocessing (encoding, deduplication, type fixes)
- Feature engineering (BMI categories, one-hot encoding)
- Feature selection using Pearson Correlation and Chi-Square tests
- Training a Linear Regression model and evaluating it with R² and Adjusted R²

---

## Dataset

**File:** `insurance.csv`

| Column | Description |
|--------|-------------|
| `age` | Age of the insured person |
| `sex` | Gender (male / female) |
| `bmi` | Body Mass Index |
| `children` | Number of dependents |
| `smoker` | Whether the person smokes (yes / no) |
| `region` | Residential region in the US |
| `charges` | Medical insurance charges (target variable) |

---

## Project Structure

```
├── EDA.ipynb          # Main notebook
├── insurance.csv      # Dataset
└── README.md
```

---

## Steps Covered in the Notebook

**1. EDA**
- Checked shape, data types, null values
- Plotted distributions for age, bmi, children, charges
- Visualized categorical columns (sex, smoker)
- Correlation heatmap

**2. Data Cleaning**
- Removed duplicate rows
- Encoded `sex` → `is_female` (0/1)
- Encoded `smoker` → `is_smoker` (0/1)
- One-hot encoded `region` (dropped first to avoid multicollinearity)

**3. Feature Engineering**
- Created `bmi_category` from BMI values (underweight / normal / overweight / obese)
- One-hot encoded `bmi_category`
- Standardized numeric features: `age`, `bmi`, `children`

**4. Feature Selection**
- Pearson Correlation for numeric features vs charges
- Chi-Square test for categorical features vs binned charges
- Final features selected: `age`, `is_female`, `bmi`, `children`, `is_smoker`, `region_southeast`, `bmi_category_obese`

**5. Model Training**
- 80/20 train-test split
- Linear Regression from scikit-learn
- Evaluated using R² and Adjusted R²

---

## How to Run

1. Clone the repo and make sure `insurance.csv` is in the same folder as the notebook

2. Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

3. Open and run the notebook:
```bash
jupyter notebook EDA.ipynb
```

---

## Results

The model was evaluated on the held-out test set (20% of data) using:

- **R² Score** — measures how much variance in charges the model explains
- **Adjusted R²** — penalizes for adding unhelpful features, more reliable than plain R²

---

## Libraries Used

- `pandas`, `numpy` — data manipulation
- `matplotlib`, `seaborn` — visualization
- `scikit-learn` — preprocessing, model training, evaluation
- `scipy` — statistical tests (Pearson, Chi-Square)

---

## Notes

- The `charges` column was binned into quartiles only for the Chi-Square test and is not used as a feature in the model
- `StandardScaler` was fit only on training data to avoid data leakage
- `drop_first=True` was used in all one-hot encoding to avoid the dummy variable trap
