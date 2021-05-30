def test_predict_dag(dag_bag):
    dag = dag_bag.dags['predict']

    dag_flow = {
        'wait-for-data': ['docker-airflow-predict'],
        'wait-for-model': ['docker-airflow-predict'],
        'docker-airflow-predict': []
    }

    for name, task in dag.task_dict.items():
        assert task.downstream_task_ids == set(dag_flow[name])
