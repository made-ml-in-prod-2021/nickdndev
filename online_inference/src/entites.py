from typing import List, Union

import numpy as np
from pydantic import BaseModel, conlist, validator

# In production this features should be in config file
FEATURES_MODELS = {'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
                   'ca', 'thal', 'id'}


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
