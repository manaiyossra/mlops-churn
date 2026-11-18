"""Déploiement planifié des flows Prefect (atelier 3, étape 5).

Lancer avec : python deploiement_prefect.py
Le script reste actif : il joue le rôle de worker et exécute les flows
selon leur planning ou quand on les déclenche manuellement.
"""

from prefect import serve
from prefect.client.schemas.schedules import CronSchedule

from pipeline_prefect import flow_all, flow_code, flow_evaluate, flow_train

TZ = "Africa/Tunis"

if __name__ == "__main__":
    all_deploy = flow_all.to_deployment(
        name="ml-pipeline-all",
        schedules=[CronSchedule(cron="0 2 * * *", timezone=TZ)],  # chaque jour à 02:00
        tags=["mlops", "full-pipeline"],
    )
    train_deploy = flow_train.to_deployment(
        name="ml-pipeline-train",
        schedules=[CronSchedule(cron="0 3 * * *", timezone=TZ)],  # chaque jour à 03:00
        tags=["mlops", "train"],
    )
    evaluate_deploy = flow_evaluate.to_deployment(
        name="ml-pipeline-evaluate",
        tags=["mlops", "evaluate"],  # pas de planning : lancement manuel uniquement
    )
    code_deploy = flow_code.to_deployment(
        name="ml-pipeline-code",
        tags=["mlops", "code"],
    )

    serve(all_deploy, train_deploy, evaluate_deploy, code_deploy)
