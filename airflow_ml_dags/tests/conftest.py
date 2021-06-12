import os
import sys

import pytest
from airflow.models import DagBag

sys.path.append('dags')


@pytest.fixture()
def dag_bag():
    os.environ["DATA_FOLDER_PATH"] = "/tmp"
    return DagBag(dag_folder='dags/', include_examples=False)
