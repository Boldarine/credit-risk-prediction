import pandas as pd
from credit_default.data import load_data, split_data, scale_data
from credit_default.models import (
    create_logistic_regression,
    create_decision_tree,
    create_random_forest,
    create_xgboost,
    create_xgboost_tuned,
    train_model
)

from credit_default.evaluation import (
    evaluate_model,
    evaluate_model_threshold,
)

from credit_default.error_analysis import analyze_errors
from credit_default.explainability import explain_model

# 1. Load data
df = load_data()

# 2. Split data into training and test sets
X_train, X_test, y_train, y_test = split_data(df)

# 3. Scale features for Logistic Regression
X_train_scaled, X_test_scaled, scaler = scale_data(
    X_train,
    X_test
)

# 4. Train Models

logistic_model = train_model(
    create_logistic_regression(),
    X_train_scaled,
    y_train
)

tree_model = train_model(
    create_decision_tree(),
    X_train,
    y_train
)

rf_model = train_model(
    create_random_forest(),
    X_train,
    y_train
)

xgb_model = train_model(
    create_xgboost(),
    X_train,
    y_train
)

xgb_tuned_model = train_model(
    create_xgboost_tuned(),
    X_train,
    y_train,
)

# 5. Evaluate models
results = [
    evaluate_model(logistic_model, X_test_scaled, y_test),
    evaluate_model(tree_model, X_test, y_test),
    evaluate_model(rf_model, X_test, y_test),
    evaluate_model(xgb_model, X_test, y_test),
]

# 6. Display results
results_df = pd.DataFrame(results)

print("\nModel evaluation results:")
print(results_df.to_string(index=False, float_format="%.4f"))

final_results = evaluate_model_threshold(
    model=xgb_tuned_model,
    X_test=X_test,
    y_test=y_test,
    threshold=0.25,
)

print("\nTuned XGBoost — final threshold 0.25:")
print(pd.DataFrame([final_results]).to_string(
    index=False, float_format="%.4f"
))

# 7. Error Analysis

error_results = analyze_errors(
    model=xgb_tuned_model,
    X_test=X_test,
    y_test=y_test,
    threshold=0.25,
)

print("\nPrediction counts:")
print(error_results["prediction_counts"])

print("\nComparison by prediction type:")
print(
    error_results["comparison_table"].to_string(
        float_format="%.2f"
    )
)

print("\nTrue positives vs false negatives:")
print(error_results["tp_fn"].to_string(float_format="%.2f"))

print("\nTrue negatives vs false positives:")
print(error_results["tn_fp"].to_string(float_format="%.2f"))

# 8. Explainability
shap_values, shap_importance = explain_model(
    model=xgb_tuned_model,
    X_test=X_test,
)

print("\nTop 10 SHAP features:")
print(
    shap_importance
    .sort_values("Mean_Abs_SHAP", ascending=False)
    .head(10)
    .to_string(index=False, float_format="%.4f")
)