"""Point d'entrée : exécute les étapes du pipeline via des arguments en ligne de commande.

Exemples :
    python main.py --prepare
    python main.py --train
    python main.py --evaluate
    python main.py --predict
    python main.py --all
"""

import argparse

from model_pipeline import (
    evaluate_model,
    load_model,
    predict,
    prepare_data,
    save_model,
    train_model,
)

DATA_PATH = "Churn_Modelling.csv"
MODEL_PATH = "classifier.joblib"

SAMPLE_CLIENT = {
    "CreditScore": 850,
    "Gender": 0,
    "Age": 43,
    "Tenure": 2,
    "Balance": 125510.82,
    "NumOfProducts": 1,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 79084.10,
}


def main():
    parser = argparse.ArgumentParser(description="Pipeline ML - Churn client")
    parser.add_argument("--data", default=DATA_PATH, help="chemin du fichier CSV")
    parser.add_argument("--model", default=MODEL_PATH, help="chemin du modèle .joblib")
    parser.add_argument("--prepare", action="store_true", help="préparer les données")
    parser.add_argument(
        "--train", action="store_true", help="entraîner et sauvegarder le modèle"
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="évaluer le modèle sauvegardé"
    )
    parser.add_argument(
        "--predict", action="store_true", help="prédire pour un client exemple"
    )
    parser.add_argument("--all", action="store_true", help="exécuter tout le pipeline")
    args = parser.parse_args()

    if not any([args.prepare, args.train, args.evaluate, args.predict, args.all]):
        parser.print_help()
        return

    if args.prepare or args.train or args.evaluate or args.all:
        x_train, x_test, y_train, y_test = prepare_data(args.data)
        print(f"Données préparées : train={x_train.shape}, test={x_test.shape}")

    if args.train or args.all:
        model = train_model(x_train, y_train)
        print("Modèle entraîné.")
        save_model(model, args.model)

    if args.evaluate or args.all:
        model = load_model(args.model)
        results = evaluate_model(model, x_test, y_test)
        print(f"Accuracy : {results['accuracy'] * 100:.2f} %")
        print("Matrice de confusion :\n", results["confusion_matrix"])
        print(results["report"])

    if args.predict or args.all:
        model = load_model(args.model)
        label = "va partir (churn)" if predict(model, SAMPLE_CLIENT) else "reste client"
        print(f"Prédiction pour le client exemple : {label}")


if __name__ == "__main__":
    main()
