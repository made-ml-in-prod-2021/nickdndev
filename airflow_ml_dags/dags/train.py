from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago

default_args = {
    "owner": "airflow_ml_dags",
    "depends_on_past": False,
    "email": ["airflow_ml_dags@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
        "train",
        default_args=default_args,
        schedule_interval="@weekly",
        start_date=days_ago(2),
) as dag:
    preprocess = DockerOperator(
        image="airflow-preprocess",
        command="--input-dir /data/raw/{{ ds }} --output-dir /data/processed/{{ ds }}",
        task_id="docker-airflow-preprocess",
        do_xcom_push=False,
        volumes=[f"/home/nickdn/Documents/made/ml_in_prod/sample/data_airflow:/data"],
    )

    split = DockerOperator(
        image="airflow-split",
        command="--input-dir /data/processed/{{ ds }}",
        task_id="docker-airflow-split",
        do_xcom_push=False,
        volumes=[f"/home/nickdn/Documents/made/ml_in_prod/sample/data_airflow:/data"],
    )

    train = DockerOperator(
        image="airflow-train",
        command="--input-dir /data/processed/{{ ds }} --output-dir /data/models/{{ ds }}",
        task_id="docker-airflow-train",
        do_xcom_push=False,
        volumes=[f"/home/nickdn/Documents/made/ml_in_prod/sample/data_airflow:/data"],
    )

    validate = DockerOperator(
        image="airflow-validate",
        command="--input-dir /data/processed/{{ ds }} --input-model-dir /data/models/{{ ds }}",
        task_id="docker-airflow-validate",
        do_xcom_push=False,
        volumes=[f"/home/nickdn/Documents/made/ml_in_prod/sample/data_airflow:/data"],
    )

    preprocess >> split >> train >> validate
