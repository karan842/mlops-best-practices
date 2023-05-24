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
    
    def __init__(self, model, experiment_name, run_name, run_metrics, run_params):
        self.model = model
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.run_metrics = run_metrics
        self.run_params = run_params
    
    def create_experiment(self):
        
        mlflow.set_tracking_uri("http://localhost:5000")    
        mlflow.set_experiment(self.experiment_name)
    
        
        with mlflow.start_run(run_name = self.run_name):
            
            if not self.run_params == None:
                for param in self.run_params:
                    mlflow.log_param(param, self.run_params[param])
            
            for metric in self.run_metrics:
                mlflow.log_metric(metric, self.run_metrics[metric])
                
            mlflow.set_tag("tag1", "Customer Churn")
             
            if isinstance(self.model, XGBClassifier):
                mlflow.xgboost.log_model(self.model, "best_model", registered_model_name="XGBoost Model")
            elif isinstance(self.model, LGBMClassifier):
                mlflow.lightgbm.log_model(self.model, "best_model",registered_model_name="LGBM Model")
            else:
                mlflow.sklearn.log_model(self.model, "best_model",registered_model_name="Sklearn Model")
                
        print(f'RUN - {self.run_name} is logged to Experiment - {self.experiment_name}')
    
    def to_production(self,version,stage:bool):
        if stage:
            client = mlflow.tracking.MlflowClient()
            model_name = self.get_registered_model_name(client)
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Production",
            )  
            
    def get_registered_model_name(self, client):
        registered_models = client.list_registered_models()
        for model in registered_models:
            if model.latest_versions[0].model_version == self.model.version:
                return model.name
        raise ValueError("Registered model not found!")        
                        
                        
