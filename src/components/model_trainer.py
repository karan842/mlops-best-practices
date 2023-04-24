import os
import sys
from dataclasses import dataclass
from collections import Counter
from sklearn.model_selection import train_test_split

from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
     f1_score, accuracy_score, roc_auc_score
)

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
                "Xgboost": XGBClassifier(),
                "Adaboost": AdaBoostClassifier(),
                "KNN": KNeighborsClassifier(),
                "Logistic Regression": LogisticRegression(),
            }
            
            model_report:dict = evaluate_models(X_train,y_train,X_valid,y_valid,models)
            
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
            
            predicted_test = best_model.predict(X_test)
            
            roc_auc_score_test = roc_auc_score(y_test, predicted_test)
            
            return(
                roc_auc_score_test
            )
            
        except Exception as e:
            
            raise CustomException(e,sys)