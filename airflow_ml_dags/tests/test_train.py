

def test_train_dag(dag_bag):
    dag = dag_bag.dags['train']

    dag_flow = {
        'wait-for-data': ['docker-airflow-preprocess'],
        'wait-for-target': ['docker-airflow-preprocess'],
        'docker-airflow-preprocess': ['docker-airflow-split'],
        'docker-airflow-split': ['docker-airflow-train'],
        'docker-airflow-train': ['docker-airflow-validate'],
        'docker-airflow-validate': []
    }
    for name, task in dag.task_dict.items():
        assert task.downstream_task_ids == set(dag_flow[name])
