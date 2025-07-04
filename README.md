
# Empowering Loan Approval Prediction 🏦🔍

A data-driven machine learning solution for predicting loan approvals based on applicant details, income, credit history, and more. This project is designed to assist financial institutions in making informed lending decisions efficiently.

---

## 📌 Table of Contents

- [About the Project](#about-the-project)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Data Preprocessing](#data-preprocessing)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [User Interaction Module](#user-interaction-module)
- [Results and Accuracy](#results-and-accuracy)
- [How to Run](#how-to-run)
- [License](#license)

---

## 📖 About the Project

This project leverages machine learning to predict whether a loan will be approved based on user inputs such as income, credit history, education, and more. It aims to empower users and financial analysts by providing accurate and explainable predictions using logistic regression, support vector machines, and random forests.

---

## 📊 Dataset

- **Source**: Loan prediction dataset (`loan_prediction.csv`)
- **Features include**:
  - Gender, Marital Status, Education
  - Applicant and Coapplicant Income
  - Loan Amount and Term
  - Credit History
  - Property Area
  - Loan Status (target variable)

---

## 💻 Technologies Used

- **Language**: Python
- **Libraries**:
  - `pandas`, `numpy` – Data handling
  - `seaborn`, `matplotlib` – Visualization
  - `scikit-learn` – ML models, preprocessing, evaluation
  - `joblib` – Model persistence

---

## 📈 Exploratory Data Analysis

- Bar plots for categorical variables like Gender, Education, Property Area
- Distribution of `Loan_Status` visualized
- Count plots grouped by `Loan_Status` for categorical features
- Null value analysis and visualization

---

## 🧹 Data Preprocessing

- Removal of irrelevant columns (e.g., `Loan_ID`)
- Null value handling using mode imputation
- Feature encoding for categorical variables
- Normalization using `StandardScaler` for numerical features

---

## 🤖 Model Training and Evaluation

Implemented and compared the performance of:

1. **Logistic Regression**
2. **Support Vector Classifier (SVM)**
3. **Decision Tree**
4. **Random Forest Classifier**

Each model was evaluated using:
- **Accuracy Score**
- **Cross-Validation**

---

## 🔍 Hyperparameter Tuning

Used **RandomizedSearchCV** to optimize:
- **Logistic Regression**: C, solver
- **SVM**: C, kernel
- **Random Forest**: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`

Visualization of tuned model accuracies using bar plots.

---

## 👤 User Interaction Module

- A dynamic Python-based questionnaire captures user input through the terminal.
- Calculates an **approximate Credit History score** based on answers.
- Converts all inputs into model-ready features.
- Predicts and displays:
  - `Loan Approved ✅`
  - `Loan Not Approved ❌`

---

## 📊 Results and Accuracy

| Model                  | Accuracy (after tuning) |
|------------------------|--------------------------|
| Logistic Regression    | ~78%                     |
| Support Vector Machine | ~77%                     |
| Random Forest          | ~81%                     |

📌 *Random Forest achieved the highest performance.*

---

## 🚀 How to Run

1. **Install dependencies:**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn joblib
   ```

2. **Run the main file** (after replacing path to your dataset):
   ```bash
   python loan_prediction.py
   ```

3. **Follow the prompts** to enter applicant details.

4. **Model Output:**
   - Displays prediction
   - Shows user data summary
   - Graphs model performances

---

## 📂 Project Structure

```
.
├── loan_prediction.csv
├── loan_prediction.py
├── loan_status_predict (saved model)
├── README.md
└── requirements.txt

## ⭐️ Give this project a star if you found it useful!
