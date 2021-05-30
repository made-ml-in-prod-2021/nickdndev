from datetime import timedelta

from airflow import DAG
from airflow.models import Variable
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
        "data_download",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=days_ago(2),
) as dag:
    data_download = DockerOperator(
        image="airflow-dataset-download",
        command="/data/raw/{{ ds }}",
        network_mode="bridge",
        task_id="docker-airflow-dataset-download",
        do_xcom_push=False,
        volumes=[f"{Variable.get('DATA_FOLDER_PATH')}:/data"],
    )
