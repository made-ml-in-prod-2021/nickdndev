from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago

from constants import DEFAULT_ARGS, DATASET_RAW_DIR, MOUNT_DATA_FOLDER

with DAG(
        "data_download",
        default_args=DEFAULT_ARGS,
        schedule_interval="@daily",
        start_date=days_ago(2),
) as dag:
    data_download = DockerOperator(
        image="airflow-dataset-download",
        command=DATASET_RAW_DIR,
        network_mode="bridge",
        task_id="docker-airflow-dataset-download",
        do_xcom_push=False,
        volumes=MOUNT_DATA_FOLDER,
    )
