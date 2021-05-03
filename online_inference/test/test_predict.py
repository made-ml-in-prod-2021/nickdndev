import os

import pandas as pd
from fastapi.testclient import TestClient
from src.app import app

def test_read_main():
    os.environ['PATH_TO_MODEL'] = '../model/model.pkl'
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == "it is entry point of our predictor"


def test_prediction():
    os.environ['PATH_TO_MODEL'] = '../model/model.pkl'

    with TestClient(app) as client:
        data_df = pd.read_csv('../data/heart.csv')
        data_df = data_df.drop('target', axis=1)

        data = data_df.values.tolist()[:15]
        features = data_df.columns.tolist()

        response = client.get("/predict/", json={"data": data, "features": features}, )
        assert response.status_code == 200
        assert response.json()[0]['diagnosis'] == 0
        assert response.json()[2]['diagnosis'] == 1
