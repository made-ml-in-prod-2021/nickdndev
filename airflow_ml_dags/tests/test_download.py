def test_data_download_dag(dag_bag):
    dag = dag_bag.dags['data_download']

    dag_flow = {
        'docker-airflow-dataset-download': [],
    }
    for name, task in dag.task_dict.items():
        assert task.downstream_task_ids == set(dag_flow[name])
