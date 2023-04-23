import os
import sys

import numpy as np
import pandas as pd
import dill
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_val_score

from collections import Counter

from src.exception import CustomException
from src.logger import logging

# save the artifacts
def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)

        CustomException(e,sys)
        
## fixing outliers
def fix_outliers(X):
    try:
        Q1 = X.quantile(0.25)
        Q3 = X.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR    
        X_outliers = X.copy()
        X_outliers[X < lower_bound] = Q1  
        X_outliers[X > upper_bound] = Q3
        return X_outliers
     
    except Exception as e:
        CustomException(e,sys)
    
    
## checking if classes are imbalance or not
def class_imbalance(y_train,thresh):
    try:
        counter = Counter(y_train)
        imbalance_percent = 100 * (1 - counter[1] / counter[0])
        imbalance_high = imbalance_percent > thresh
        return imbalance_high 
      
    except Exception as e:
        CustomException(e,sys)

## handling class imbalance        
def handling_class_imbalance(X,y,thresh):
    try:
        if class_imbalance(y,thresh):
            logging.info("Target class is imbalanced.")
            
            smote = SMOTE()
            X_balanced, y_balanced = smote.fit_resample(X,y)
            
            logging.info("Imbalancing of the data has been fixed")
            return X_balanced, y_balanced
        
        else:
            logging.info("Target class is already imbalanced.")
            return X,y
    
    except Exception as e:
        CustomException(e,sys)
        
## K-Fold cross validation
def cross_validate_model(model,X,y,n_folds):
    try:
        scores = cross_val_score(model,X,y,cv=n_folds,scoring='roc_auc')
        return scores.mean()
        logging.info("Performed cross validation")
    
    except Exception as e:
        raise CustomException(e,sys)
    
# evaluate the model    
def evaluate_models(X_train,y_train,X_test,y_test,models):
    try:
        
        report = {}
        
        for i in range(len(list(models))):
            
            model = list(models.values())[i]
            
            model.fit(X_train,y_train)
            
            y_train_pred = model.predict(X_train)
            
            y_test_pred = model.predict(X_test)
            
            train_model_score = roc_auc_score(y_train, y_train_pred)
            
            test_model_score = roc_auc_score(y_test, y_test_pred)
            
            report[list(models.keys())[i]] = test_model_score
            
        return report
    
    except Exception as e:
        raise CustomException(e,sys)