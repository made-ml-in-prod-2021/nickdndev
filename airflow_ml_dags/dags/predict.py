from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.dates import days_ago

from constants import DEFAULT_ARGS, MOUNT_DATA_FOLDER, DATASET_RAW_DATA_FILE_NAME, DATASET_RAW_DIR, MODEL_FILE_NAME, \
    MODELS_DIR, DATASET_PREDICTION_DIR

with DAG(
        "predict",
        default_args=DEFAULT_ARGS,
        schedule_interval="@daily",
        start_date=days_ago(8),
) as dag:
    wait_for_data = FileSensor(
        task_id='wait-for-data',
        poke_interval=5,
        retries=5,
        filepath='/'.join([DATASET_RAW_DIR, DATASET_RAW_DATA_FILE_NAME])
    )

    wait_for_model = FileSensor(
        task_id='wait-for-model',
        poke_interval=5,
        retries=5,
        filepath='/'.join([MODELS_DIR, MODEL_FILE_NAME])
    )

    predict = DockerOperator(
        image="airflow-predict",
        command=f"--input-dir {DATASET_RAW_DIR} --input-model-dir {MODELS_DIR} --output-dir {DATASET_PREDICTION_DIR}",
        task_id="docker-airflow-predict",
        do_xcom_push=False,
        volumes=MOUNT_DATA_FOLDER,
    )

    [wait_for_data, wait_for_model] >> predict
