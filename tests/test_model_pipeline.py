"""Tests unitaires des fonctions de model_pipeline.py."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_pipeline import (  # noqa: E402
    evaluate_model,
    load_model,
    prepare_data,
    save_model,
    train_model,
)

DATA_PATH = "Churn_Modelling.csv"


def test_prepare_data_shapes():
    """Le split doit donner 8000 lignes de train, 2000 de test et 9 colonnes."""
    x_train, x_test, y_train, y_test = prepare_data(DATA_PATH)
    assert x_train.shape == (8000, 9)
    assert x_test.shape == (2000, 9)
    assert len(y_train) == 8000 and len(y_test) == 2000


def test_prepare_data_columns():
    """Les colonnes inutiles sont supprimées et Gender est encodé en 0/1."""
    x_train, _, _, _ = prepare_data(DATA_PATH)
    for col in ["RowNumber", "CustomerId", "Surname", "Geography", "Exited"]:
        assert col not in x_train.columns
    assert set(x_train["Gender"].unique()) <= {0, 1}


def test_train_and_evaluate():
    """Le modèle entraîné doit dépasser 80 % d'accuracy."""
    x_train, x_test, y_train, y_test = prepare_data(DATA_PATH)
    model = train_model(x_train, y_train)
    results = evaluate_model(model, x_test, y_test)
    assert results["accuracy"] > 0.80


def test_save_and_load(tmp_path):
    """Un modèle sauvegardé puis rechargé donne les mêmes prédictions."""
    x_train, x_test, y_train, _ = prepare_data(DATA_PATH)
    model = train_model(x_train, y_train)
    path = tmp_path / "model.joblib"
    save_model(model, path)
    loaded = load_model(path)
    assert (loaded.predict(x_test) == model.predict(x_test)).all()
