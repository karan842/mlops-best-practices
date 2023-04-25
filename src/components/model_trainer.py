import os
import sys
import json
from dataclasses import dataclass
from collections import Counter
from sklearn.model_selection import train_test_split

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
import mlflow
from mlflow.tracking import MlflowClient

from src.exception import CustomException
from src.logger import logging
from src.utils import (
    save_object, evaluate_models, cross_validate_model
)


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_array,test_array):
        
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (train_array[:,:-1], train_array[:,-1], test_array[:,:-1],test_array[:,-1])
            
            ## validation data
            X_train, X_valid, y_train, y_valid = train_test_split(X_train,y_train,test_size=0.2,random_state=42)
            
           
            models = {
                "Random Forest": RandomForestClassifier(),
                "XGBoost": XGBClassifier(),
                "LGBM": LGBMClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "Logistic Regression": LogisticRegression(),
            }
            
            # loading parameters for tuning 
            with open('artifacts/params.json') as params_file:
                params = json.load(params_file)
            
            model_report:dict = evaluate_models(X_train,y_train,X_valid,y_valid,
                                                models,params)
            
            best_model_score = max(sorted(model_report.values()))
            
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException("No best model found!")
            logging.info("Best found model on both training and testing dataset.")
            
            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
            
            # log best model and its parameters with MLflow 
            # with mlflow.start_run(run_name="Best Model"):
            #     # log hyperparameters
            #     for key, value in best_model.get_params().items():
            #         mlflow.log_param(key,value)
                    
            #     # train and log model artifact
            #     if isinstance(best_model,XGBClassifier):
            #         best_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
            #         mlflow.xgboost.log_model(best_model,"model")
            #     elif isinstance(best_model, CatBoostClassifier):
            #         best_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],verbose=False)
            #         mlflow.catboost.log_model(best_model,"model")
            #     else:
            #         best_model.fit(X_train, y_train)
            #         mlflow.sklearn.log_model(best_model,"model")
            
            # Log metrics
            predicted_test = best_model.predict(X_test)
            accuracy_score_test = accuracy_score(y_test, predicted_test)
            precision_score_test = precision_score(y_test, predicted_test)
            recall_score_test = recall_score(y_test, predicted_test)
            f1_score_test = f1_score(y_test, predicted_test)
            roc_auc_score_test = roc_auc_score(y_test, predicted_test)
            
            metrics = {'Best Model':best_model_name,
                       'Accuracy':accuracy_score_test,
                       'Precision':precision_score_test,
                       'Recall':recall_score_test,
                       'ROC-AUC':roc_auc_score_test}
            
            # for key, value in metrics.items()
            
            with open('artifacts/metrics.json','w') as metrics_file:
                json.dump(metrics,metrics_file,indent=4)
            logging.info("Metrics saved")
            
            return(
                metrics
            )
            
        except Exception as e:
            
            raise CustomException(e,sys)