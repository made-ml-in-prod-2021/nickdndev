from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.dates import days_ago

from constants import DEFAULT_ARGS, DATASET_RAW_DATA_FILE_NAME, DATASET_RAW_DIR, \
    DATASET_RAW_TARGET_FILE_NAME, MOUNT_DATA_FOLDER, DATASET_PROCESSED_DIR, MODELS_DIR

with DAG(
        "train",
        default_args=DEFAULT_ARGS,
        schedule_interval="@weekly",
        start_date=days_ago(2),
) as dag:
    wait_for_data = FileSensor(
        task_id='wait-for-data',
        poke_interval=5,
        retries=5,
        filepath='/'.join([DATASET_RAW_DIR, DATASET_RAW_DATA_FILE_NAME])
    )

    wait_for_target = FileSensor(
        task_id='wait-for-target',
        poke_interval=5,
        retries=5,
        filepath='/'.join([DATASET_RAW_DIR, DATASET_RAW_TARGET_FILE_NAME])
    )

    preprocess = DockerOperator(
        image="airflow-preprocess",
        command=f"--input-dir {DATASET_RAW_DIR} --output-dir {DATASET_PROCESSED_DIR} ",
        task_id="docker-airflow-preprocess",
        do_xcom_push=False,
        volumes=MOUNT_DATA_FOLDER
    )

    split = DockerOperator(
        image="airflow-split",
        command=f"--input-dir {DATASET_PROCESSED_DIR}",
        task_id="docker-airflow-split",
        do_xcom_push=False,
        volumes=MOUNT_DATA_FOLDER,
    )

    train = DockerOperator(
        image="airflow-train",
        command=f"--input-dir {DATASET_PROCESSED_DIR} --output-dir {MODELS_DIR}",
        task_id="docker-airflow-train",
        do_xcom_push=False,
        volumes=MOUNT_DATA_FOLDER,
    )

    validate = DockerOperator(
        image="airflow-validate",
        command=f"--input-dir {DATASET_PROCESSED_DIR} --input-model-dir {MODELS_DIR}",
        task_id="docker-airflow-validate",
        do_xcom_push=False,
        volumes=MOUNT_DATA_FOLDER,
    )

    [wait_for_data, wait_for_target] >> preprocess >> split >> train >> validate
