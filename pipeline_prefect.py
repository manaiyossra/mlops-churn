"""Pipeline MLOps orchestré avec Prefect (atelier 3).

Exemples :
    python pipeline_prefect.py --flow code
    python pipeline_prefect.py --flow train
    python pipeline_prefect.py --flow evaluate
    python pipeline_prefect.py --flow all
"""

import argparse
import subprocess  # nosec B404 - commandes fixes, sans entrée utilisateur
import sys

from prefect import flow, get_run_logger, task

import model_pipeline as mp

DATA_PATH = "Churn_Modelling.csv"
MODEL_PATH = "classifier.joblib"
PROJECT_FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]


def _run(cmd):
    """Exécute une commande système et lève une erreur si elle échoue."""
    logger = get_run_logger()
    logger.info("Commande : %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)  # nosec B603
    if result.stdout:
        logger.info(result.stdout)
    if result.returncode != 0:
        logger.error(result.stderr or result.stdout)
        raise RuntimeError(f"Échec de la commande : {' '.join(cmd)}")


# ---------------------------------------------------------------- tasks : code
@task(name="install-dependencies")
@task(name="git-pull")
def git_pull():
    """Récupère la dernière version du projet depuis le dépôt Git distant."""
    _run(["git", "pull"])
def install_dependencies():
    """Installe les dépendances listées dans requirements.txt."""
    _run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])


@task(name="format-code")
def format_code():
    """Formate automatiquement le code avec black."""
    _run([sys.executable, "-m", "black", *PROJECT_FILES, "tests/"])


@task(name="code-quality")
def code_quality():
    """Vérifie la qualité du code avec flake8."""
    _run([sys.executable, "-m", "flake8", "--max-line-length=100", *PROJECT_FILES])


@task(name="code-security")
def code_security():
    """Analyse la sécurité du code avec bandit (sévérité moyenne et plus)."""
    _run([sys.executable, "-m", "bandit", "-ll", *PROJECT_FILES])


@task(name="unit-tests")
def unit_tests():
    """Exécute les tests unitaires avec pytest."""
    _run([sys.executable, "-m", "pytest", "tests/", "-q"])


# ---------------------------------------------------------- tasks : data/modèle
@task(name="prepare-data")
def prepare_data():
    """Charge et prépare les données (appelle model_pipeline.prepare_data)."""
    x_train, x_test, y_train, y_test = mp.prepare_data(DATA_PATH)
    get_run_logger().info("train=%s, test=%s", x_train.shape, x_test.shape)
    return x_train, x_test, y_train, y_test


@task(name="train-model")
def train_model(x_train, y_train):
    """Entraîne le modèle."""
    return mp.train_model(x_train, y_train)


@task(name="save-model")
def save_model(model):
    """Sauvegarde le modèle entraîné."""
    mp.save_model(model, MODEL_PATH)


@task(name="load-model")
def load_model():
    """Charge le modèle sauvegardé."""
    return mp.load_model(MODEL_PATH)


@task(name="evaluate-model")
def evaluate_model(model, x_test, y_test):
    """Évalue le modèle et affiche les métriques."""
    results = mp.evaluate_model(model, x_test, y_test)
    logger = get_run_logger()
    logger.info("Accuracy : %.2f %%", results["accuracy"] * 100)
    logger.info("Matrice de confusion :\n%s", results["confusion_matrix"])
    return results["accuracy"]


@task(name="predict")
def predict(model):
    """Prédit le churn pour un client exemple."""
    client = {
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
    label = "va partir (churn)" if mp.predict(model, client) else "reste client"
    get_run_logger().info("Prédiction pour le client exemple : %s", label)


# ----------------------------------------------------------------------- flows
@flow(name="code")
def flow_code():
    """Formatage, qualité, sécurité et tests unitaires du code."""
    format_code()
    code_quality()
    code_security()
    unit_tests()


@flow(name="train")
def flow_train():
    """Préparation des données puis entraînement et sauvegarde du modèle."""
    x_train, _, y_train, _ = prepare_data()
    model = train_model(x_train, y_train)
    save_model(model)


@flow(name="evaluate")
def flow_evaluate():
    """Chargement du modèle sauvegardé et évaluation."""
    _, x_test, _, y_test = prepare_data()
    model = load_model()
    evaluate_model(model, x_test, y_test)


@flow(name="all")
def flow_all():
    """Pipeline complet : git, dépendances, code, données, entraînement, évaluation."""
    git_pull()
    install_dependencies()
    flow_code()
    x_train, x_test, y_train, y_test = prepare_data()
    model = train_model(x_train, y_train)
    save_model(model)
    evaluate_model(model, x_test, y_test)
    predict(model)


FLOWS = {
    "code": flow_code,
    "train": flow_train,
    "entrainement": flow_train,
    "evaluate": flow_evaluate,
    "all": flow_all,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline MLOps avec Prefect")
    parser.add_argument("--flow", required=True, choices=FLOWS.keys())
    args = parser.parse_args()
    FLOWS[args.flow]()
