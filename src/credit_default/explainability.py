
import pandas as pd
import matplotlib.pyplot as plt
import shap


def explain_model(model, X_test):
    # 1. Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    # 2. Plot global feature impact
    shap.plots.beeswarm(shap_values)

    # 3. Calculate mean absolute SHAP importance
    shap_importance = pd.DataFrame({
        "Feature": X_test.columns,
        "Mean_Abs_SHAP": abs(shap_values.values).mean(axis=0),
    })

    # 4. Select the 10 most important features
    top_10_shap = (
        shap_importance
        .sort_values("Mean_Abs_SHAP", ascending=False)
        .head(10)
    )

    # 5. Plot feature importance
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.barh(
        top_10_shap["Feature"],
        top_10_shap["Mean_Abs_SHAP"],
    )

    ax.invert_yaxis()
    ax.set_title("Top 10 Features by Mean Absolute SHAP Value")
    ax.set_xlabel("Mean Absolute SHAP Value")
    ax.set_ylabel("Feature")

    plt.tight_layout()
    plt.show()

    return shap_values, shap_importance
