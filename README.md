# 💰 Insurance Cost Prediction using Machine Learning

This project predicts **annual medical insurance charges** using machine learning regression models based on demographic and health-related features.

It is designed as a **complete, step-by-step machine learning project**, easy for beginners to understand and professional enough for interviews and portfolio review.

---

## 📌 Project Overview

Health insurance companies need accurate cost predictions to:
- Offer fair insurance premiums
- Reduce financial risk
- Improve transparency for customers

This project builds an **end-to-end machine learning regression pipeline** to estimate insurance charges for individuals.

After training and comparing multiple models, **Random Forest Regressor** was selected as the **final model** due to its **stable performance, good accuracy, and strong generalization**.

---

## 🎯 Problem Statement

The goal of this project is to **predict annual medical insurance charges** for a person using demographic and health-related features.

Since the target variable (**charges**) is a continuous numerical value, this is a **Regression Problem**.

---

## 🧾 Dataset Information

- **Dataset Name:** `insurance_Data.csv`
- **Target Variable:** `charges`

### 📊 Feature Description

| Feature | Description |
|-------|------------|
| age | Age of the individual |
| sex | Gender (male / female) |
| bmi | Body Mass Index |
| children | Number of dependent children |
| smoker | Smoking status (yes / no) |
| region | Residential region |
| charges | Annual insurance charges (Target Variable) |

---

## 🧪 Step-by-Step Machine Learning Workflow

This project follows a **standard machine learning process**, explained clearly below.

---

### 🔹 Step 1: Exploratory Data Analysis (EDA)

EDA is performed to understand the dataset before model building.

Actions performed:
- Checked dataset shape and data types
- Verified missing values (no missing values found)
- Analyzed distributions of numerical features
- Identified outliers using boxplots
- Studied relationships between features and target
- Performed correlation analysis

Key insights:
- Smoking status has the strongest impact on insurance charges
- Age and BMI significantly influence medical costs
- Insurance charges are right-skewed

---

### 🔹 Step 2: Data Preprocessing & Feature Engineering

The dataset is prepared for machine learning.

Steps performed:
- Separated features (`X`) and target (`y`)
- Encoded categorical variables using **OrdinalEncoder**
- Scaled numerical variables using **StandardScaler**
- Handled feature outliers (BMI capped)
- Target variable outliers were **not removed** (they represent real costs)
- Used **ColumnTransformer** for column-wise preprocessing

---

### 🔹 Step 3: Train–Test Split

The dataset is split into:
- **80% Training data**
- **20% Testing data**

Purpose:
- Train the model on one part of the data
- Evaluate performance on unseen data
- Prevent overfitting

---

### 🔹 Step 4: Model Building

Multiple regression models were trained using **machine learning pipelines** to ensure consistent preprocessing.

Models trained:
- Linear Regression
- K-Nearest Neighbors Regressor
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

---

### 🔹 Step 5: Model Evaluation

All models were evaluated using standard regression metrics:
- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**
- **R² Score**

These metrics help compare prediction accuracy and error magnitude.

---

### 🔹 Step 6: Hyperparameter Tuning

Hyperparameter tuning was performed using **RandomizedSearchCV**.

Important notes:
- A reduced search space was used to save computation time
- In some cases, default parameters performed similarly or slightly better
- This indicates the baseline model was already well-optimized

---

### ⭐ Step 7: Final Model Selection

**Final Model Selected:** **Random Forest Regressor**

Reasons:
- Stable and high R² score
- Low RMSE
- Strong generalization on test data
- Robust to non-linear relationships and outliers

---

## 📈 Final Model Performance (Random Forest)

| Metric | Performance |
|------|------------|
| MAE | Low |
| RMSE | Low |
| R² Score | High |

(Exact metric values are available in the notebook output.)

---

## 🚀 Prediction System

The final trained model:
- Accepts user inputs (age, BMI, smoking status, etc.)
- Applies preprocessing automatically using pipeline
- Predicts annual insurance charges accurately
- Model is saved as a **pickle file** for reuse or deployment

---

## ▶️ How to Run This Project (Step-by-Step)

1. Clone the repository:
   ```bash
   git clone <your-repo-link>
