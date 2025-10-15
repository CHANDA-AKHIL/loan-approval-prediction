# loan-approval-prediction
Classification project predicting loan approval using Logistic Regression on Kaggle Playground S4E10 dataset.
## Overview
This project is based on the Kaggle Playground Series - Season 4, Episode 10: Loan Approval Prediction. The goal is to predict whether an applicant is approved for a loan using synthetically generated tabular data. The competition ran from October 1 to November 1, 2024, with submissions evaluated using the area under the ROC curve (ROC-AUC).

Key skills demonstrated:
- Data preprocessing (handling missing values, encoding categoricals)
- Feature scaling with StandardScaler
- Classification using Logistic Regression
- Evaluation with ROC-AUC
- Visualization of distributions and model performance

Developed as part of my BTech in Computer Science to practice classification techniques.

Citation: Walter Reade and Ashley Chow. Loan Approval Prediction. https://kaggle.com/competitions/playground-series-s4e10, 2024. Kaggle.

## Dataset
- **Source**: [Kaggle Playground S4E10](https://kaggle.com/competitions/playground-series-s4e10/data)
- **Description**: Synthetic tabular data with features like income, credit history, and loan amount. Train set includes loan_status (0=Rejected, 1=Approved).
- **Preprocessing**:
  - Filled missing categorical values with mode, numerical with median.
  - One-hot encoded categorical features.

Note: Datasets not included due to size. Download from Kaggle.
## Methodology
1. **Data Preparation**: Load CSVs, handle missing values, encode categoricals.
2. **Modeling**: Train Logistic Regression with scaled features.
3. **Evaluation**: Compute ROC-AUC on validation set (20% split).
4. **Visualizations**: Loan status distribution, correlation heatmap, ROC curve.
5. **Prediction**: Generate test set probabilities and submission file.

Results: Achieved ROC-AUC ~0.7 (adjust based on your run); key features include credit history.
