## Comments

**Target:** `default payment next month` — 1 = default, 0 = no default.

The dataset contains 30,000 rows and 25 columns. All columns have `int64` data types, and no missing values were found.

**Variable descriptions:**
- `LIMIT_BAL`: Amount of credit granted.
- `SEX`: 1 = male, 2 = female.
- `EDUCATION`: 1 = graduate school, 2 = university, 3 = high school, 4 = others.
- `MARRIAGE`: 1 = married, 2 = single, 3 = others.
- `AGE`: Age in years.
- `PAY_0` to `PAY_6`: Repayment status, with `PAY_0` referring to September 2005 and `PAY_2` to August 2005; subsequent variables refer to earlier months.
- `BILL_AMT1` to `BILL_AMT6`: Bill statement amounts.
- `PAY_AMT1` to `PAY_AMT6`: Previous payment amounts.

**Target Distribution**

The target variable is imbalanced:
- No default (0): 92.46%
- Default (1): 7.54%

**EDUCATION:** The dataset contains 345 records with undocumented category values (0, 5, and 6). These values were mapped to category 4 ("Others"), following the documented category mapping. This decision was applied before rerunning the modeling workflow. The treatment is a practical assumption, as the original meaning of these values is unknown.

**MARRIAGE** contains values 0, that is not included in the documented category mapping (1-3)
54 records

**BILL_AMT** Negative values were identified across all periods. The available documentation does not provide enough information to determine whether they represent valid values or inconsistencies. These values will be kept for now, and the need for any treatment will be evaluated later.

**Feature Scaling:** During baseline modeling, Logistic Regression produced a convergence warning. StandardScaler was applied to the training features, and the same transformation was applied to the test features. Scaling resolved the convergence issue. Tree-based models were trained without feature scaling.

**Baseline Comparison (Initial Modeling):** Random Forest achieved the highest F1-score (0.4662), PR-AUC (0.5316), and Accuracy (0.8145) among the four baseline models. Logistic Regression achieved the highest Precision (0.6868) and Specificity (0.9690), but had substantially lower Recall (0.2396). XGBoost achieved the highest ROC-AUC (0.7596), while Random Forest achieved a higher PR-AUC (0.5316). Given the class imbalance and the importance of identifying positive cases, PR-AUC, Recall, Precision, and F1-score were prioritized during subsequent model evaluation.

**Final Model Comparison:** After the EDUCATION treatment, Random Forest and XGBoost showed similar overall performance. The models were evaluated at different classification thresholds selected for their respective precision-recall trade-offs: 0.30 for Random Forest and 0.25 for XGBoost.

Random Forest achieved higher Precision (0.5503 vs. 0.4975), F1-score (0.5410 vs. 0.5408), and Specificity (0.8765 vs. 0.8301). XGBoost achieved higher Recall (0.5923 vs. 0.5320), identifying 80 more default cases and producing 80 fewer false negatives, at the cost of 217 additional false positives. ROC-AUC and PR-AUC were very similar between the models.

XGBoost was retained as the candidate model because identifying a larger proportion of default cases was the primary objective of this project.

### Error Analysis
False negatives tended to have higher credit limits, higher previous payment amounts, and less evident historical payment delinquency than true positives. The median `PAY_0` value was 0 for false negatives, compared with 2 for true positives.

False positives tended to show more evidence of recent payment delinquency, lower credit limits, and lower previous payment amounts than true negatives. The median `PAY_0` value was 1 for false positives, compared with 0 for true negatives.

These patterns suggest that the model identifies clear historical delinquency signals relatively well but has difficulty distinguishing some future defaulters whose observed behavior appears less risky from non-defaulters who exhibit similar risk signals. These findings are descriptive and do not establish causal relationships.


## Conclusion

### Research Question
The objective was to predict which customers would default on their credit card payment in the following month.

### Model Results
XGBoost was retained as the final candidate model because identifying default cases was a priority.

- **Threshold:** 0.25
- **Recall:** 0.5923
- **Precision:** 0.4975
- **F1-score:** 0.5408
- **PR-AUC:** 0.5554
- **ROC-AUC:** 0.7772
- **Specificity:** 0.8301

The selected threshold prioritizes identifying a larger proportion of default cases, accepting a higher number of false positives as a trade-off.

### Main Findings
- Recent payment history (`PAY_0`, `PAY_2`, `PAY_3`, etc.) was substantially more important to the model than demographic variables.
- `PAY_0` was the most important feature, accounting for approximately 42.6% of the model's total feature importance, followed by `PAY_2` at 9.4%.
- Payment history and payment-related variables played a major role in the model's predictions.
- The model identified customers with clear historical delinquency patterns relatively well.
- False negatives tended to have higher credit limits, higher payment amounts, and less evident previous payment delinquency than true positives.
- False positives tended to show stronger delinquency signals, lower credit limits, and lower payment amounts than true negatives.
- The results highlight a trade-off between identifying more default cases and generating more false positives.

### Limitations
- The dataset represents a specific historical population and may not generalize to other customer populations or time periods.
- No true temporal validation was performed.
- The classification threshold was selected through exploratory analysis on the test set and should therefore be considered a project-level choice rather than an unbiased estimate of future performance.
- The model identifies predictive patterns but does not establish causal relationships.
- The available variables may not capture all factors associated with future default.

### Final Assessment
The project demonstrates that machine learning can identify meaningful patterns associated with future credit card default. XGBoost was retained as the candidate model because it achieved higher Recall at the selected threshold, aligning with the project's priority of identifying a larger proportion of default cases. This came at the cost of lower Precision and Specificity compared with Random Forest. The model still misses a substantial proportion of actual defaults, indicating that historical payment behavior alone is not sufficient to perfectly distinguish future defaulters from non-defaulters.