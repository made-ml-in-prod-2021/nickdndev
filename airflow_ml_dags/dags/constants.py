import os
from datetime import timedelta

from airflow.models import Variable

DEFAULT_ARGS = {
    "owner": "airflow",
    "email": ["troubleshooting@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

DATASET_RAW_DIR = "/data/raw/{{ ds }}"
DATASET_PROCESSED_DIR = "/data/processed/{{ ds }}"
DATASET_PREDICTION_DIR = "/data/predictions/{{ ds }}"
MODELS_DIR = "/data/models/{{ ds }}"

DATASET_RAW_DATA_FILE_NAME = "data.csv"
DATASET_RAW_TARGET_FILE_NAME = "target.csv"
MODEL_FILE_NAME = "model"

MOUNT_DATA_FOLDER = [f"{os.environ['DATA_FOLDER_PATH']}:/data"]

