import os
from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago

default_args = {
    "owner": "airflow_ml_dags",
    "email": ["airflow_ml_dags@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
        "11_docker",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=days_ago(5),
) as dag:
    download = DockerOperator(
        image="mikhailmar/airflow_ml_dags-download",
        command="/data/raw/{{ ds }}",
        network_mode="bridge",
        task_id="docker-airflow_ml_dags-download",
        do_xcom_push=False,
        volumes=["/Users/mikhail.maryufich/PycharmProjects/airflow_examples/data:/data"]
    )

    preprocess = DockerOperator(
        image="mikhailmar/airflow_ml_dags-preprocess",
        command="--input-dir /data/raw/{{ ds }} --output-dir /data/processed/{{ ds }}",
        task_id="docker-airflow_ml_dags-preprocess",
        do_xcom_push=False,
        volumes=["/Users/mikhail.maryufich/PycharmProjects/airflow_examples/data:/data"]
    )

    predict = DockerOperator(
        image="mikhailmar/airflow_ml_dags-predict",
        command="--input-dir /data/processed/{{ ds }} --output-dir /data/predicted/{{ ds }}",
        task_id="docker-airflow_ml_dags-predict",
        do_xcom_push=False,
        volumes=["/Users/mikhail.maryufich/PycharmProjects/airflow_examples/data:/data"]
    )

    download >> preprocess >> predict
