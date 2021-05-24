from datetime import timedelta

from airflow.models import Variable
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago

from airflow import DAG

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}
with DAG(
        "data_download",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=days_ago(3),
) as dag:
    data_download = DockerOperator(
        image="airflow-dataset-download",
        command=f"--output-dir /data/raw/{{ ds }}",
        network_mode="bridge",
        task_id="docker-airflow-dataset-download",
        do_xcom_push=False,
        volumes=[f"{Variable.get('data_path')}:/data"],
    )
