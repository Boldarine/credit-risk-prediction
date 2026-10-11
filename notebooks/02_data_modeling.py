# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    classification_report,
    average_precision_score
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

# Function to evalute metrics for all tested models
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)

    return {
        "Model": model.__class__.__name__,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "Specificity": specificity,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }

# Load raw dataset
df = pd.read_excel("../data/raw/credit_card_clients.xls", header=1)

# Check EDUCATION values before treatment
print("Before treatment:")
print(df["EDUCATION"].value_counts().sort_index())

# Map undocumented education categories to "Others" (4)
df["EDUCATION"] = df["EDUCATION"].replace({
    0: 4,
    5: 4,
    6: 4
})

# Check EDUCATION values after treatment
print("\nAfter treatment:")
print(df["EDUCATION"].value_counts().sort_index())

# The logistic model will be my baseline for the project
X = df.drop(columns=["default payment next month", "ID"])
y = df["default payment next month"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logistic_model = LogisticRegression (
    random_state=42,
    max_iter=1000
)

logistic_model.fit(X_train_scaled, y_train)

y_pred = logistic_model.predict(X_test_scaled)
y_proba = logistic_model.predict_proba(X_test_scaled)[:,1]

# %%
# Validating if the scaler worked
print(X_train_scaled.mean(axis=0))
print(X_train_scaled.std(axis=0))
print(logistic_model.n_iter_)
# %%
# Results for logistic model
logistic_results = evaluate_model(
    logistic_model,
    X_test_scaled,
    y_test
)

print(logistic_results)

# %%
####################### DECISION TREE #######################

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)

# Results for tree model
tree_results = evaluate_model(
    tree,
    X_test,
    y_test
)

print(tree_results)

# %%
####################### RANDON FOREST #######################

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=0)
rf.fit(X_train, y_train)

# Results for random forest
rf_results = evaluate_model(
    rf,
    X_test,
    y_test
)

print(rf_results)

# %%
####################### XGBOOST       #######################
from xgboost import XGBClassifier

xgb = XGBClassifier(random_state=42)
xgb.fit(X_train, y_train)

# Results for random forest
xgb_results = evaluate_model(
    xgb,
    X_test,
    y_test
)

print(xgb_results)


# %%
### Tabela formatada de testes
results = [
    logistic_results,
    tree_results,
    rf_results,
    xgb_results]

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False, float_format="%.4f"))
# %%
#####################################################################################

## Tuning Random

from sklearn.model_selection import RandomizedSearchCV

rf_tunned = RandomForestClassifier(random_state=42)

param_distributions = {
    "n_estimators": [200, 400, 600, 800],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 5, 10],
    "max_features": ["sqrt", "log2", None]
}

rf_search = RandomizedSearchCV(
    estimator=rf_tunned,
    param_distributions=param_distributions,
    n_iter=50,
    cv=5,
    scoring="average_precision",
    random_state=42,
    n_jobs=-1
)

rf_search.fit(X_train, y_train)
# %%
print("Best parameters:")
print(rf_search.best_params_)

print("\nBest CV PR-AUC:")
print(rf_search.best_score_)
# %%
rf_tuned_results = evaluate_model(
    rf_search.best_estimator_,
    X_test,
    y_test
)

print(rf_tuned_results)
# %%
### Comparing Random Forest
results_rf = [
    rf_results,
    rf_tuned_results
    ]

results_rf_df = pd.DataFrame(results_rf)
print(results_rf_df.to_string(index=False, float_format="%.4f"))
# %%
import numpy as np

# Best Random Forest from RandomizedSearchCV
rf_tunned = rf_search.best_estimator_

# Probabilities for the positive class
y_proba = rf_tunned.predict_proba(X_test)[:, 1]

thresholds = np.arange(0.20, 0.61, 0.05)

threshold_results = []

for threshold in thresholds:
    y_pred_threshold = (y_proba >= threshold).astype(int)

    precision = precision_score(y_test, y_pred_threshold)
    recall = recall_score(y_test, y_pred_threshold)
    f1 = f1_score(y_test, y_pred_threshold)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred_threshold
    ).ravel()

    specificity = tn / (tn + fp)

    threshold_results.append({
        "Threshold": threshold,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Specificity": specificity,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    })

threshold_df = pd.DataFrame(threshold_results)

print(
    threshold_df.to_string(
        index=False,
        float_format="%.4f"
    )
)
# %%
## Tuning XGBoost

xgb = XGBClassifier(
    random_state=42
)

param_distributions = {
    "n_estimators": [100, 200, 400, 600],
    "max_depth": [3, 5, 7, 10],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "min_child_weight": [1, 3, 5, 10],
}

xgb_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_distributions,
    n_iter=50,
    cv=5,
    scoring="average_precision",
    random_state=42,
    n_jobs=-1
)

xgb_search.fit(X_train, y_train)

# %%
print("Best parameters:")
print(xgb_search.best_params_)

print("\nBest CV PR-AUC:")
print(xgb_search.best_score_)
# %%
xgb_tuned_results = evaluate_model(
    xgb_search.best_estimator_,
    X_test,
    y_test
)

print(xgb_tuned_results)
# %%
### Comparing XGBoost
results_xgb = [
    xgb_results,
    xgb_tuned_results
    ]

results_xgb_df = pd.DataFrame(results_xgb)
print(results_xgb_df.to_string(index=False, float_format="%.4f"))
# %%
# Best XGBoost from RandomizedSearchCV
xgb_tunned = xgb_search.best_estimator_

# Probabilities for the positive class
y_proba = xgb_tunned.predict_proba(X_test)[:, 1]

thresholds = np.arange(0.20, 0.61, 0.05)

threshold_results = []

for threshold in thresholds:
    y_pred_threshold = (y_proba >= threshold).astype(int)

    precision = precision_score(y_test, y_pred_threshold)
    recall = recall_score(y_test, y_pred_threshold)
    f1 = f1_score(y_test, y_pred_threshold)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred_threshold
    ).ravel()

    specificity = tn / (tn + fp)

    threshold_results.append({
        "Threshold": threshold,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Specificity": specificity,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    })

threshold_df = pd.DataFrame(threshold_results)

print(
    threshold_df.to_string(
        index=False,
        float_format="%.4f"
    )
)
# %%
def evaluate_model_threshold(model, X_test, y_test, threshold):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred
    ).ravel()

    specificity = tn / (tn + fp)

    return {
        "Model": model.__class__.__name__,
        "Threshold": threshold,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "Specificity": specificity,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }

rf_final_results = evaluate_model_threshold(
    rf_search.best_estimator_,
    X_test,
    y_test,
    threshold=0.30
)

xgb_final_results = evaluate_model_threshold(
    xgb_search.best_estimator_,
    X_test,
    y_test,
    threshold=0.25
)


final_results = [
    rf_final_results,
    xgb_final_results
]

final_results_df = pd.DataFrame(final_results)

print(
    final_results_df.to_string(
        index=False,
        float_format="%.4f"
    )
)
# %%
### Feature Importance - XGBoost
xgb_model = xgb_search.best_estimator_
xgb_model.feature_importances_
len(xgb_model.feature_importances_)

# %%
feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": xgb_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    "Importance",
    ascending=False
)

print(feature_importance)
# %%
### SHAP
import shap

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer(X_test)

shap.plots.beeswarm(shap_values)
# %%
import matplotlib.pyplot as plt

# Calculate mean absolute SHAP value for each feature
shap_importance = pd.DataFrame({
    "Feature": X_test.columns,
    "Mean_Abs_SHAP": abs(shap_values.values).mean(axis=0)
})

# Select the 10 most important features
top_10_shap = shap_importance.sort_values(
    "Mean_Abs_SHAP",
    ascending=False
).head(10)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

ax.barh(
    top_10_shap["Feature"],
    top_10_shap["Mean_Abs_SHAP"]
)

ax.invert_yaxis()

ax.set_title("Top 10 Features by Mean Absolute SHAP Value")
ax.set_xlabel("Mean Absolute SHAP Value")
ax.set_ylabel("Feature")

plt.tight_layout()
plt.show()
# %%
#####################################################################################
### Error Analisis

import numpy as np

# Get predicted probabilities from the final XGBoost model
y_proba = xgb_model.predict_proba(X_test)[:, 1]

# Apply the selected threshold
threshold = 0.25
y_pred = (y_proba >= threshold).astype(int)

# Create a copy of the test set
error_analysis = X_test.copy()

# Add actual and predicted values
error_analysis["Actual"] = y_test.values
error_analysis["Predicted"] = y_pred
error_analysis["Probability"] = y_proba

# Classify each prediction
error_analysis["Prediction_Type"] = np.select(
    [
        (error_analysis["Actual"] == 1) & (error_analysis["Predicted"] == 1),
        (error_analysis["Actual"] == 0) & (error_analysis["Predicted"] == 0),
        (error_analysis["Actual"] == 0) & (error_analysis["Predicted"] == 1),
        (error_analysis["Actual"] == 1) & (error_analysis["Predicted"] == 0)
    ],
    [
        "TP",
        "TN",
        "FP",
        "FN"
    ],
    default="UNKNOWN"
)

# Count each prediction type
prediction_counts = (
    error_analysis["Prediction_Type"]
    .value_counts()
    .reindex(["TP", "TN", "FP", "FN"])
)

print(prediction_counts)
# %%
features_to_compare = [
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "LIMIT_BAL",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6"
]

comparison_table = (
    error_analysis
    .groupby("Prediction_Type")[features_to_compare]
    .agg(["mean", "median"])
    .reindex(["TP", "TN", "FP", "FN"])
)

print(comparison_table.to_string(float_format="%.2f"))
# %%
# Compare true positives and false negatives
tp_fn = (
    error_analysis[
        error_analysis["Prediction_Type"].isin(["TP", "FN"])
    ]
    .groupby("Prediction_Type")[features_to_compare]
    .agg(["mean", "median"])
    .reindex(["TP", "FN"])
)

print(tp_fn.to_string(float_format="%.2f"))
# %%
# Compare true negatives and false positives
tn_fp = (
    error_analysis[
        error_analysis["Prediction_Type"].isin(["TN", "FP"])
    ]
    .groupby("Prediction_Type")[features_to_compare]
    .agg(["mean", "median"])
    .reindex(["TN", "FP"])
)

print(tn_fp.to_string(float_format="%.2f"))
# %%
print("Best parameters:")
print(xgb_search.best_params_)

print("\nBest CV PR-AUC:")
print(xgb_search.best_score_)
# %%
