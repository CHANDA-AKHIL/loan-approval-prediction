# ================================================
# Loan Approval Prediction - Logistic Regression
# Author: [Chanda Akhil]
# Description: This script predicts loan approval status using Logistic Regression on Kaggle's synthetic dataset.
# Includes data preprocessing, model training, ROC-AUC evaluation, and visualizations.
# Dependencies: pandas, numpy, scikit-learn, matplotlib, seaborn
# ================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
import warnings
warnings.filterwarnings("ignore")

# ================================================
# 1. Load Data
# ================================================
# Load datasets from 'data/' folder; download from https://kaggle.com/competitions/playground-series-s4e10/data
try:
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
except FileNotFoundError as e:
    print(f"Error: {e}. Please download datasets and place in 'data/' folder.")
    exit(1)

print("Data loaded successfully!")
print("Train shape:", train.shape)
print("Test shape:", test.shape)

# ================================================
# 2. Prepare Features and Target
# ================================================
target = "loan_status"
id_col = "id"

X = train.drop(columns=[target, id_col])
y = train[target]
X_test = test.drop(columns=[id_col])

# ================================================
# 3. Handle Missing Values
# ================================================
for col in X.columns:
    if X[col].dtype == "object":
        X[col].fillna(X[col].mode()[0], inplace=True)
        X_test[col].fillna(X[col].mode()[0], inplace=True)
    else:
        X[col].fillna(X[col].median(), inplace=True)
        X_test[col].fillna(X[col].median(), inplace=True)

# ================================================
# 4. Encode Categorical Variables
# ================================================
X = pd.get_dummies(X, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

# Align columns between train and test
X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)
if X.shape[1] != X_test.shape[1]:
    print("Warning: Feature mismatch between train and test sets.")

# ================================================
# 5. Train-Test Split and Feature Scaling
# ================================================
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

# ================================================
# 6. Train Logistic Regression Model
# ================================================
model = LogisticRegression(max_iter=1000, solver='lbfgs', n_jobs=-1)
model.fit(X_train_scaled, y_train)

# ================================================
# 7. Evaluate on Validation Set
# ================================================
y_pred_prob = model.predict_proba(X_valid_scaled)[:, 1]
roc_auc = roc_auc_score(y_valid, y_pred_prob)
print(f" Validation ROC-AUC: {roc_auc:.4f}")

# ================================================
# 8. Predict on Test Set and Create Submission File
# ================================================
test_pred = model.predict_proba(X_test_scaled)[:, 1]

submission = pd.DataFrame({
    id_col: test[id_col],
    target: test_pred
})
submission.to_csv('submission.csv', index=False)
print("Submission file created successfully: submission.csv")

# ================================================
# 9. Visualizations
# ================================================
# (A) Loan Status Distribution
plt.figure(figsize=(6, 5))
sns.countplot(x=target, data=train, palette=["#2E86AB", "#A23B72"])
plt.title("Loan Status Distribution", fontsize=16, fontweight='bold')
plt.xlabel("Loan Status (0=Rejected, 1=Approved)")
plt.ylabel("Count")
for container in plt.gca().containers:
    plt.gca().bar_label(container)
plt.tight_layout()
plt.savefig('visualizations/loan_status_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# (B) Correlation Heatmap
plt.figure(figsize=(10, 8))
numeric_cols = train.select_dtypes(include=np.number)
correlation = numeric_cols.corr()
sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, center=0, linewidths=0.5, square=True)
plt.title("Correlation Heatmap", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations/correlation_heatmap_loan.png', dpi=300, bbox_inches='tight')
plt.show()

# (C) ROC Curve
fpr, tpr, thresholds = roc_curve(y_valid, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.3f})", color='#2E86AB', linewidth=3)
plt.plot([0, 1], [0, 1], '--', color='gray', linewidth=2, label='Random Classifier')
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("ROC Curve - Loan Approval Prediction", fontsize=16, fontweight='bold')
plt.legend(fontsize=11, loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/roc_curve_loan.png', dpi=300, bbox_inches='tight')
plt.show()


print("All visualizations saved!")
