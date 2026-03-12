# Credit Card Fraud Detection

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-green)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-red)
![Imbalanced Data](https://img.shields.io/badge/SMOTE-Imbalanced--Learning-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

End-to-End Machine Learning Pipeline for Detecting Fraudulent Credit Card Transactions using SMOTE, Random Forest, and XGBoost.

This project builds a complete machine learning workflow to identify fraudulent credit card transactions from an extremely imbalanced dataset. The pipeline includes data preprocessing, class imbalance handling using SMOTE, training multiple models, evaluating performance, and saving the best model.

---

## Table of Contents

- Problem Statement  
- Objective  
- Dataset  
- Technologies Used  
- Machine Learning Models  
- Project Workflow  
- Evaluation Metrics  
- Model Performance  
- Project Structure  
- Results  
- How to Run  
- Future Improvements  
- Author  

---

## Problem Statement

Credit card fraud detection is a challenging machine learning problem because fraudulent transactions are extremely rare compared to normal transactions.

In most financial transaction datasets, fraud cases represent less than **1% of total transactions**, making this a **highly imbalanced classification problem**.

Machine learning models must be trained carefully to detect these rare fraud cases while minimizing false positives.

---

## Objective

The main objectives of this project are:

- Build an end-to-end machine learning pipeline for fraud detection  
- Handle severe class imbalance using SMOTE  
- Train multiple machine learning models  
- Compare model performance using classification metrics  
- Save the best performing model  

---

## Dataset

The dataset used in this project is the **Credit Card Fraud Detection Dataset** from Kaggle.

Dataset Source:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Dataset Characteristics:

- Total Transactions: 284,807  
- Fraud Transactions: 492  
- Features: 31 columns  
- Highly imbalanced dataset  

Target Variable:

```
Class
```

```
0 → Normal Transaction
1 → Fraud Transaction
```

---

## Technologies Used

### Programming
Python

### Data Science Libraries
Pandas  
NumPy  
Matplotlib  
Seaborn  

### Machine Learning
Scikit-learn  
XGBoost  
imbalanced-learn (SMOTE)

### Tools
Git  
GitHub  
Joblib  

---

## Machine Learning Models

This project compares multiple classification models:

- Logistic Regression (Baseline Model)  
- Random Forest (Ensemble Model)  
- XGBoost (Gradient Boosting Model)  

These models help identify patterns that distinguish fraudulent transactions from legitimate ones.

---

## Project Workflow

1. Data Loading  
2. Exploratory Data Analysis (EDA)  
3. Class Distribution Analysis  
4. Train-Test Split  
5. SMOTE Oversampling  
6. Model Training  
7. Model Evaluation  
8. Model Comparison  
9. Best Model Selection  
10. Model Saving  

---

## Evaluation Metrics

Because fraud detection involves imbalanced data, accuracy alone is not sufficient.

The models are evaluated using:

- Accuracy  
- Precision  
- Recall  
- F1 Score  
- Confusion Matrix  
- Classification Report  

Special focus is given to **Recall and F1-score for the fraud class**.

---

## Model Performance

| Model | Precision | Recall | F1 Score |
|------|-----------|--------|---------|
| Logistic Regression | ~0.54 | ~0.88 | ~0.67 |
| Random Forest | ~0.92 | ~0.91 | ~0.91 |
| XGBoost | ~0.94 | ~0.93 | ~0.93 |

*Performance values may vary slightly depending on dataset split.*

---

## Project Structure

```
credit-card-fraud-detection/
│
├── data/                    # Dataset folder
│
├── models/                  # Saved trained models
│
├── src/                     # Source code
│   ├── data_preprocessing.py
│   ├── train_model.py
│   └── evaluate_model.py
│
├── notebooks/               # Future experimentation notebooks
│
├── requirements.txt         # Project dependencies
│
└── README.md                # Project documentation
```

---

## Results

The machine learning pipeline successfully detects fraudulent transactions by addressing severe class imbalance using SMOTE.

Ensemble models such as **Random Forest and XGBoost** significantly improve fraud detection performance compared to the baseline model.

---

## How to Run the Project

Install dependencies

```
pip install -r requirements.txt
```

Run preprocessing

```
python src/data_preprocessing.py
```

Train the model

```
python src/train_model.py
```

Evaluate the model

```
python src/evaluate_model.py
```

---

## Future Improvements

- Hyperparameter tuning using GridSearchCV or RandomizedSearchCV  
- Feature scaling and feature selection  
- Visualization of model comparison  
- Experiment tracking using MLflow  
- Deploying the model using FastAPI  
- Containerizing the project using Docker  

---

## Author

Bhushan Patil  
AI / Machine Learning Engineer  
Pune, Maharashtra, India