from datetime import timedelta

from airflow import DAG
from airflow.models import Variable
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor
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
    wait_for_data = FileSensor(
        task_id='wait-for-data',
        poke_interval=5,
        retries=5,
        filepath="data/raw/{{ ds }}/data.csv"
    )

    wait_for_target = FileSensor(
        task_id='wait-for-target',
        poke_interval=5,
        retries=5,
        filepath="data/raw/{{ ds }}/target.csv"
    )

    preprocess = DockerOperator(
        image="airflow-preprocess",
        command="--input-dir /data/raw/{{ ds }} --output-dir /data/processed/{{ ds }}",
        task_id="docker-airflow-preprocess",
        do_xcom_push=False,
        volumes=[f"{Variable.get('DATA_FOLDER_PATH')}:/data"],
    )

    split = DockerOperator(
        image="airflow-split",
        command="--input-dir /data/processed/{{ ds }}",
        task_id="docker-airflow-split",
        do_xcom_push=False,
        volumes=[f"{Variable.get('DATA_FOLDER_PATH')}:/data"],
    )

    train = DockerOperator(
        image="airflow-train",
        command="--input-dir /data/processed/{{ ds }} --output-dir /data/models/{{ ds }}",
        task_id="docker-airflow-train",
        do_xcom_push=False,
        volumes=[f"{Variable.get('DATA_FOLDER_PATH')}:/data"],
    )

    validate = DockerOperator(
        image="airflow-validate",
        command="--input-dir /data/processed/{{ ds }} --input-model-dir /data/models/{{ ds }}",
        task_id="docker-airflow-validate",
        do_xcom_push=False,
        volumes=[f"{Variable.get('DATA_FOLDER_PATH')}:/data"],
    )

    [wait_for_data, wait_for_target] >> preprocess >> split >> train >> validate
