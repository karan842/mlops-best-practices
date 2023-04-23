import os
import sys
from dataclasses import dataclass
from collections import Counter

from imblearn.over_sampling import SMOTE

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
            
            models = {
                "Random Forest": RandomForestClassifier(),
                "Xgboost": XGBClassifier(),
                "Adaboost": AdaBoostClassifier(),
                "KNN": KNeighborsClassifier(),
                "Logistic Regression": LogisticRegression(),
            }
            
            cv_results = {}
            
            for name, model in models.items():
                cv_score = cross_validate_model(model,X_train,y_train,5)
                cv_results[name] = cv_score
                
            best_model_name = max(cv_results,key=cv_results.get)
            best_model = models[best_model_name]
            
            if cv_results[best_model_name] < 0.6:
                raise CustomException("No best model found!")
            logging.info("Best found model on both training and testing dataset.")
            
            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
            
            best_model.fit(X_train,y_train)
            
            predicted = best_model.predict(X_test)
            
            roc_auc_score_ = roc_auc_score(y_test,predicted)
            
            return {best_model_name:roc_auc_score_}
            
        except Exception as e:
            
            raise CustomException(e,sys)