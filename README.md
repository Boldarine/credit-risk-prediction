# Credit Default Prediction

## Objective
Develop a machine learning model to predict whether a credit card client will default on payment next month.

## Dataset
The project uses the Default of Credit Card Clients dataset, containing 30,000 records and 23 explanatory features, plus an ID and the target variable.

- Target: `default payment next month`
- Class 0: No default
- Class 1: Default
- Class distribution: approximately 92.46% non-default and 7.54% default

## Methodology
- Data loading and understanding
- Feature preparation and train/test split
- Baseline comparison: Logistic Regression, Decision Tree, Random Forest, and XGBoost
- Hyperparameter tuning with RandomizedSearchCV
- Evaluation using Accuracy, Precision, Recall, F1-score, ROC-AUC, PR-AUC, and Specificity
- Error analysis and SHAP explainability

## Final Model
**Tuned XGBoost**, using a classification threshold of `0.25`.

| Metric | Result |
|---|---:|
| Accuracy | 0.7775 |
| Precision | 0.4975 |
| Recall | 0.5923 |
| F1-score | 0.5408 |
| ROC-AUC | 0.7772 |
| PR-AUC | 0.5554 |
| Specificity | 0.8301 |

The lower threshold prioritizes identifying more potential defaulters, increasing recall at the cost of more false positives.

## Explainability
SHAP analysis identified the following among the most influential features:

- `PAY_0`
- `LIMIT_BAL`
- `BILL_AMT1`
- `PAY_AMT2`
- `PAY_AMT1`

`PAY_0`, representing the most recent repayment status, had the highest mean absolute SHAP value.

## Limitations
- The dataset has a substantial class imbalance.
- The threshold was selected during exploratory analysis using the test set, which may introduce optimistic bias. Ideally, threshold selection should use a validation set, leaving the test set untouched for final evaluation.
- Predictive associations do not imply causation.

## Project Structure
- `notebooks/01_data_understanding.py`: data understanding
- `notebooks/02_data_modeling.py`: experiments and model selection
- `src/credit_default/`: reusable data, modeling, evaluation, error analysis, and explainability functions
- `run_experiment.py`: runs the end-to-end experiment

## Running the Project
Install dependencies using `uv`:

```bash
uv sync
```

Run the experiment:

```bash
uv run python run_experiment.py
```

Place the dataset at `data/raw/credit_card_clients.xls` before running the project.