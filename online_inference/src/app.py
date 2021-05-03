import logging
import os
import pickle
from typing import List, Union, Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, conlist
from starlette.responses import PlainTextResponse

logger = logging.getLogger(__name__)


def load_object(path: str) -> dict:
    with open(path, "rb") as f:
        model = pickle.load(f)
        return model


class DiagnosisRequest(BaseModel):
    data: List[conlist(Union[float, str], min_items=10, max_items=20)]
    features: List[str]


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
