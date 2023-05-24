import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient

from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
     f1_score, accuracy_score, roc_auc_score, precision_score, recall_score
)

from src.exception import CustomException
from src.logger import logging


class ExperimentTracking:
    
    def __init__(self, experiment_name, run_name):
        self.experiment_name = experiment_name
        self.run_name = run_name
    
    def create_experiment(self, model, run_metrics, run_params):
        
        mlflow.set_experiment(self.experiment_name)
        
        with mlflow.start_run(run_name = self.run_name):
            
            if not run_params == None:
                for param in run_params:
                    mlflow.log_param(param, run_params[param])
            
            for metric in run_metrics:
                mlflow.log_metric(metric, run_metrics[metric])
                
            mlflow.set_tag("tag1", "Customer Churn")
             
            if isinstance(model, XGBClassifier):
                mlflow.xgboost.log_model(model, "best_model")
            else:
                mlflow.sklearn.log_model(model, "best_model")
                
        print(f'RUN - {self.run_name} is logged to Experiment - {self.experiment_name}')
                        
                        
                        
