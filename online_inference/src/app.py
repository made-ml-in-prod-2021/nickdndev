import logging
import os
import pickle
from typing import List, Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.responses import PlainTextResponse

from entites import DiagnosisResponse, DiagnosisRequest

logger = logging.getLogger(__name__)


def load_object(path: str) -> dict:
    with open(path, "rb") as f:
        model = pickle.load(f)
        return model


model: Optional[dict] = None


def make_predict(data: List, features: List[str], model: dict, ) -> List[DiagnosisResponse]:
    data = pd.DataFrame(data, columns=features)
    ids = data['id']
    features = data.drop(['id'], axis=1)

    transformer = model["transformer"]
    classifier = model["classifier"]

    transformed_features = transformer.transform(features)
    predicts = classifier.predict(transformed_features)

    return [
        DiagnosisResponse(id=id, diagnosis=int(diagnosis)) for id, diagnosis in zip(ids, predicts)
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
