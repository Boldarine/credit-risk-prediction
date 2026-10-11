
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def create_logistic_regression():
    return LogisticRegression(
        random_state=42,
        max_iter=1000
    )


def create_decision_tree():
    return DecisionTreeClassifier(
        random_state=0
    )


def create_random_forest():
    return RandomForestClassifier(
        random_state=0
    )


def create_xgboost():
    return XGBClassifier(
        random_state=42
    )

def create_xgboost_tuned():
    return XGBClassifier(
        subsample=0.7,
        n_estimators=600,
        min_child_weight=1,
        max_depth=7,
        learning_rate=0.01,
        colsample_bytree=1.0,
        random_state=42,
    )

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model