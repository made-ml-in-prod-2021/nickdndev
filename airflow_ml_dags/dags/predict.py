from datetime import timedelta

from airflow import DAG
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
        "predict",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=days_ago(8),
) as dag:
    wait_for_data = FileSensor(
        task_id='wait-for-data',
        poke_interval=5,
        retries=5,
        filepath="data/raw/{{ ds }}/data.csv"
    )

    wait_for_model = FileSensor(
        task_id='wait-for-model',
        poke_interval=5,
        retries=5,
        filepath="/data/models/{{ ds }}/model"
    )

    predict = DockerOperator(
        image="airflow-predict",
        command="--input-dir /data/raw/{{ ds }} --input-model-dir /data/models/{{ ds }} --output-dir /data/predictions/{{ ds }}",
        task_id="docker-airflow-predict",
        do_xcom_push=False,
        volumes=[f"/home/nickdn/Documents/made/ml_in_prod/sample/data_airflow:/data"],
    )

    [wait_for_data, wait_for_model] >> predict
