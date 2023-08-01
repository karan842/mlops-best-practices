import os
import sys
from dotenv import load_dotenv
load_dotenv()
project_home_path = os.environ.get('PROJECT_HOME_PATH')
sys.path.append(project_home_path)
import json
from datetime import datetime
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

from src.tracking.mlflow_tracking import ExperimentTracking
from src.exception import CustomException
from src.logger import logging
from src.utils import (
    save_object, evaluate_models, cross_validate_model, get_metrics
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
                # "XGBoost": XGBClassifier(),
                # "LGBM": LGBMClassifier(),
                # "AdaBoost": AdaBoostClassifier(),
                # "Logistic Regression": LogisticRegression(),
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
            best_model_params = best_model.get_params()
            
            if best_model_score < 0.6:
                raise CustomException("No best model found!")
            logging.info("Best found model on both training and testing dataset.")
            
            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
        
            # Log metrics
            predicted_test = best_model.predict(X_test)
            evaluation_metrices = get_metrics(y_test, predicted_test)
            best_model_artifacts = {'Best Model':best_model_name}
            best_model_artifacts.update(evaluation_metrices)
            
            # mlflow tracking
            
            experiment_name = 'customer-churn-prediction-experiment'+str(datetime.now().strftime("%d-%m-%Y"))

            run_name = 'customer-churn-prediction'+str(datetime.now().strftime("%d-%m-%Y"))
            
            exp_track = ExperimentTracking(
                model = best_model,
                experiment_name = experiment_name, 
                run_name = run_name,
                run_metrics = evaluation_metrices,
                run_params = best_model_params
            )
            
            exp_track.create_experiment()
            # exp_track.to_production(version=1,stage="Production")
            
            with open('artifacts/metrics.json','w') as metrics_file:
                json.dump(best_model_artifacts,metrics_file,indent=4)
            logging.info("Best Model Metrics saved")
            
            return(
                best_model_artifacts
            )
            
        except Exception as e:
            
            raise CustomException(e,sys)