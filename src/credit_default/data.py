import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "credit_card_clients.xls"

def load_data():
    df = pd.read_excel(DATA_PATH, header=1)

    df["EDUCATION"] = df["EDUCATION"].replace({
        0: 4,
        5: 4,
        6: 4
    })

    return df

def split_data(df):
    X = df.drop(columns=["default payment next month", "ID"])
    y = df["default payment next month"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test

def scale_data(X_train, X_test):

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler

