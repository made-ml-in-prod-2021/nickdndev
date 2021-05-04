import logging
import os
import pickle
from typing import List, Union, Optional
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, conlist, validator
from starlette.responses import PlainTextResponse

logger = logging.getLogger(__name__)

# In production this features should be in config file
FEATURES_MODELS = {'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
                   'ca', 'thal'}


def load_object(path: str) -> dict:
    with open(path, "rb") as f:
        model = pickle.load(f)
        return model


class DiagnosisRequest(BaseModel):
    data: List[conlist(Union[float, str], min_items=1, max_items=20)]
    features: List[str]

    @validator('features')
    def validate_model_features(cls, features):
        if not set(features).issuperset(FEATURES_MODELS):
            raise ValueError(f'Invalid features! Valid features are: {FEATURES_MODELS}')
        return features

    @validator('data')
    def validate_number_data_columns_and_features(cls, data):
        if np.array(data).shape[1] != len(FEATURES_MODELS):
            raise ValueError(f'Invalid columns number for data! Valid numbers are: {len(FEATURES_MODELS)}')
        return data


class DiagnosisResponse(BaseModel):
    id: str
    diagnosis: int


model: Optional[dict] = None


def make_predict(data: List, features: List[str], model: dict, ) -> List[DiagnosisResponse]:
    data = pd.DataFrame(data, columns=features)

    transformer = model["transformer"]
    classifier = model["classifier"]

    features = transformer.transform(data)
    predicts = classifier.predict(features)

    return [
        DiagnosisResponse(id=idx, diagnosis=int(diagnosis)) for idx, diagnosis in enumerate(predicts)
    ]


app = FastAPI()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return PlainTextResponse(str(exc), status_code=400)


@app.get("/")
def main():
    return "it is entry point of our predictor"


@app.on_event("startup")
def load_model():
    logger.info(f"Loading model...")
    global model
    model_path = os.getenv("PATH_TO_MODEL")
    if model_path is None:
        err = f"PATH_TO_MODEL {model_path} is None"
        logger.error(err)
        raise RuntimeError(err)
    model = load_object(model_path)

    logger.info(f"Model is ready...")


@app.get("/healthz")
def health() -> bool:
    return not (model is None)


@app.get("/predict/", response_model=List[DiagnosisResponse])
def predict(request: DiagnosisRequest):
    return make_predict(request.data, request.features, model)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=os.getenv("PORT", 8000))
