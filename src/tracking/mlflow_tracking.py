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
                mlflow.xgboost.log_model(self.model, "best_model", registered_model_name="XGBoost")
            elif isinstance(self.model, LGBMClassifier):
                mlflow.lightgbm.log_model(self.model, "best_model",registered_model_name="LGBM")
            else:
                mlflow.sklearn.log_model(self.model, "best_model",registered_model_name="Sklearn")
                
        print(f'RUN - {self.run_name} is logged to Experiment - {self.experiment_name}')
    
    def to_production(self, version:int, stage: str):
        if stage:
            if isinstance(self.model, XGBClassifier):
                model_name = "XGBoost"
            elif isinstance(self.model, LGBMClassifier):
                model_name = "LGBM"
            else:
                model_name = "Sklearn"
            
            client = mlflow.tracking.MlflowClient()
            model_version = client.get_latest_versions(name=model_name, stages=[stage])[0].version
            client.transition_model_version_stage(
                name=model_name,
                version=model_version,
                stage=stage,
            )
            print(f"Model {model_name} version {model_version} is moved to stage '{stage}'.")
            
    # def get_registered_model_name(self, client):
    #     model_type = self.model.__class__.__name__
    #     registered_model_name = f"{model_type} Model"
    #     return registered_model_name