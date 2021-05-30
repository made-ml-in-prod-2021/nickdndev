import sys
from unittest import mock

import pytest
from airflow.models import DagBag

sys.path.append('dags')


@pytest.fixture()
def dag_bag():
    with mock.patch.dict('os.environ', AIRFLOW_VAR_DATA_FOLDER_PATH="env-value"):
        return DagBag(dag_folder='dags/', include_examples=False)
