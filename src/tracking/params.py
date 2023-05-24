import os
import numpy as np
import pandas as pdsaq  
import sys
import json
import matplotlib.pyplot as plt
import warnings
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
     f1_score, accuracy_score, roc_auc_score, precision_score, recall_score
)
import mlflow

from src.exception import CustomException
from src.logger import logging
from src.utils import (
    save_object, evaluate_models, cross_validate_model
)

def save_params_to_yaml():
    params = {
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 20],
            'min_samples_split': [2, 5, 10],
        },
        'XGBoost': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.1, 0.3, 0.5],
        },
        'LGBM': {
            'n_estimators': [100, 500, 1000],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.3],
            'num_leaves': [31, 63, 127],
        },
        'AdaBoost': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.1, 0.5, 1.0],
            'algorithm': ['SAMME', 'SAMME.R'],
        },
        'Logistic Regression': {
            'penalty': ['l1', 'l2'],
            'C': [0.1, 1.0, 10.0],
            'solver': ['liblinear'],
        }
    }

    with open('artifacts/params.json','w') as params_file:
        json.dump(params,params_file,indent=4)