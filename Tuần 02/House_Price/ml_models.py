"""
Machine-learning interface for exactly MODEL_01..MODEL_05.
Input: processed numpy arrays.
Public:
- build_model(model_id)
- train_one_model(model_id, X_train, y_train) -> model, training_time_seconds
- predict_price(model, X) -> numpy.ndarray

TODO TV2: tune only reasonable hyperparameters and record experiments.

"""


import time
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)
from xgboost import XGBRegressor


# BUILD MODEL
MODEL_BUILDERS = {
    "MODEL_01": lambda: LinearRegression(),
    "MODEL_02": lambda: DecisionTreeRegressor(
        random_state=42
    ),
    "MODEL_03": lambda: RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ),
    "MODEL_04": lambda: GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.03,
        random_state=42
    ),
    "MODEL_05": lambda: XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        random_state=42
    ),
}


def build_model(model_id):
    return MODEL_BUILDERS[model_id]()



# TRAIN MODEL
def train_one_model(model_id, X_train, y_train):
    # TODO TV2:
    # build model
    # fit
    # đo training time
    # return model + time

    model = build_model(model_id)
    
    start_time = time.perf_counter()
    
    model.fit(X_train, y_train)

    end_time = time.perf_counter()
    
    training_time_seconds = end_time - start_time
    
    return model, training_time_seconds

# PREDICT PRICE
def predict_price(model, X):
    predictions = model.predict(X)
    return np.asarray(predictions)