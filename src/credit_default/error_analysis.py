
import numpy as np


FEATURES_TO_COMPARE = [
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
    "PAY_AMT6",
]


def analyze_errors(model, X_test, y_test, threshold=0.25):
    # 1. Generate probabilities and predictions
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    # 2. Create the error analysis dataset
    error_analysis = X_test.copy()
    error_analysis["Actual"] = y_test.values
    error_analysis["Predicted"] = y_pred
    error_analysis["Probability"] = y_proba

    # 3. Classify each prediction
    error_analysis["Prediction_Type"] = np.select(
        [
            (error_analysis["Actual"] == 1)
            & (error_analysis["Predicted"] == 1),

            (error_analysis["Actual"] == 0)
            & (error_analysis["Predicted"] == 0),

            (error_analysis["Actual"] == 0)
            & (error_analysis["Predicted"] == 1),

            (error_analysis["Actual"] == 1)
            & (error_analysis["Predicted"] == 0),
        ],
        ["TP", "TN", "FP", "FN"],
        default="UNKNOWN",
    )

    # 4. Count prediction types
    prediction_counts = (
        error_analysis["Prediction_Type"]
        .value_counts()
        .reindex(["TP", "TN", "FP", "FN"], fill_value=0)
    )

    # 5. Compare all prediction groups
    comparison_table = (
        error_analysis
        .groupby("Prediction_Type")[FEATURES_TO_COMPARE]
        .agg(["mean", "median"])
        .reindex(["TP", "TN", "FP", "FN"])
    )

    # 6. Compare true positives and false negatives
    tp_fn = (
        error_analysis[
            error_analysis["Prediction_Type"].isin(["TP", "FN"])
        ]
        .groupby("Prediction_Type")[FEATURES_TO_COMPARE]
        .agg(["mean", "median"])
        .reindex(["TP", "FN"])
    )

    # 7. Compare true negatives and false positives
    tn_fp = (
        error_analysis[
            error_analysis["Prediction_Type"].isin(["TN", "FP"])
        ]
        .groupby("Prediction_Type")[FEATURES_TO_COMPARE]
        .agg(["mean", "median"])
        .reindex(["TN", "FP"])
    )

    return {
        "error_analysis": error_analysis,
        "prediction_counts": prediction_counts,
        "comparison_table": comparison_table,
        "tp_fn": tp_fn,
        "tn_fp": tn_fp,
    }
