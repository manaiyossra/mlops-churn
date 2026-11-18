"""Pipeline ML modulaire pour la prédiction du churn client (Churn_Modelling.csv)."""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

TARGET = "Exited"
COLUMNS_TO_DROP = ["RowNumber", "CustomerId", "Surname", "Geography"]


def prepare_data(data_path, test_size=0.2, random_state=1):
    """Charge le CSV, nettoie et encode les données, puis les découpe en train/test.

    Args:
        data_path (str): chemin du fichier CSV.
        test_size (float): proportion des données réservée au test.
        random_state (int): graine pour un découpage reproductible.

    Returns:
        tuple: (x_train, x_test, y_train, y_test)
    """
    df = pd.read_csv(data_path)

    if df.isna().sum().sum() > 0:
        df = df.dropna()
    df = df.drop_duplicates()

    df["Gender"] = LabelEncoder().fit_transform(df["Gender"])  # Female=0, Male=1
    df = df.drop(columns=COLUMNS_TO_DROP)

    x = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(x, y, test_size=test_size, random_state=random_state)


def train_model(x_train, y_train, n_estimators=100, random_state=42):
    """Entraîne un RandomForestClassifier et le renvoie."""
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(x_train, y_train)
    return model


def evaluate_model(model, x_test, y_test):
    """Calcule l'accuracy, la matrice de confusion et le rapport de classification.

    Returns:
        dict: {"accuracy", "confusion_matrix", "report"}
    """
    y_pred = model.predict(x_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "report": classification_report(y_test, y_pred),
    }


def save_model(model, path="classifier.joblib"):
    """Sauvegarde le modèle entraîné avec joblib."""
    joblib.dump(model, path)
    print(f"Modèle sauvegardé dans {path}")


def load_model(path="classifier.joblib"):
    """Charge un modèle sauvegardé avec joblib."""
    return joblib.load(path)


def predict(model, features):
    """Prédit le churn pour un client.

    Args:
        features (dict): valeurs des colonnes utilisées à l'entraînement.

    Returns:
        int: 1 si le client risque de partir, 0 sinon.
    """
    x = pd.DataFrame([features])[model.feature_names_in_]
    return int(model.predict(x)[0])
