# IEEE-CIS Fraud Detection

This project applies data mining and machine learning techniques to detect fraudulent online transactions using the **IEEE-CIS Fraud Detection dataset**. The dataset is high-dimensional, contains extensive missing values, and is highly imbalanced, making it a realistic real-world fraud detection problem.

## Project Objectives
We aimed to answer the following research questions:

1. Can transaction metadata and identity-related features improve fraud detection compared to using transaction amount alone?
2. Which feature categories contribute most to fraud prediction?
3. Which machine learning models perform best under extreme class imbalance?
4. How do preprocessing strategies impact model performance?

## Dataset
Source: IEEE-CIS Fraud Detection Dataset (Kaggle)

- Transactions: **590,540**
- Features: **434**
- Target variable: `isFraud` (binary classification)
- Fraud rate: ~**3.5%**

The dataset contains transaction-level attributes, identity/device information, and anonymized behavioral features (`V1–V339`).

## Workflow
### 1. Exploratory Data Analysis (EDA)
- Verified strong class imbalance
- Analyzed feature distributions and fraud patterns
- Identified extreme missingness in identity and distance-related columns

### 2. Preprocessing
- Dropped features with >90% missing values
- Median imputation for numerical features
- Filled missing categorical values with `"Missing"`
- One-hot encoding for categorical variables (`drop_first=True`)
- Feature engineering:
  - `TransactionAmt_log = log(1 + TransactionAmt)`
  - Time-based features from `TransactionDT` (hours/days)
- Stratified train/test split (80/20)

### 3. Modeling
Models trained and evaluated:
- Decision Tree
- Random Forest
- XGBoost
- Baseline Logistic Regression using `TransactionAmt` only

Class imbalance was handled using cost-sensitive learning (`class_weight="balanced"` and `scale_pos_weight`), without oversampling or undersampling.

## Evaluation Metrics
Due to imbalance, model performance was evaluated using:
- ROC-AUC
- PR-AUC
- Precision / Recall / F1-score

## Results Summary

| Model | ROC-AUC | PR-AUC |
|------|---------|--------|
| **XGBoost** | **0.9230** | **0.6131** |
| Random Forest | 0.9005 | 0.5692 |
| Decision Tree | 0.8639 | 0.4164 |
| Baseline (Amount Only) | 0.5004 | 0.0391 |

## Key Findings
- Transaction amount alone is not sufficient for fraud detection.
- Identity and behavioral metadata significantly improve prediction quality.
- XGBoost achieved the best overall performance in terms of ROC-AUC and PR-AUC.
- Fraud is pattern-based and linked to contextual/behavioral signals rather than random anomalies.


## Tools and Libraries
- Python (Pandas, NumPy)
- Scikit-learn
- XGBoost
- Matplotlib
- VS Code

## Authors
- Prakriti Kattel
- Nguyen Uyen
