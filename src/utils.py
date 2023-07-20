import os
import sys
from dotenv import load_dotenv
load_dotenv()
project_home_path = os.environ.get('PROJECT_HOME_PATH')
sys.path.append(project_home_path)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dill
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_validate, GridSearchCV, RandomizedSearchCV

from collections import Counter

from src.exception import CustomException
from src.logger import logging

from configure import configure

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
        quantile1 = configure["data_transformation"]["outliers"]["quantile1"]
        quantile2 = configure["data_transformation"]["outliers"]["quantile2"]
        Q1 = X.quantile(quantile1)
        Q3 = X.quantile(quantile2)
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
        cv_results = cross_validate(model,X,y,cv=n_folds,scoring='accuracy',return_estimator=True)
        best_estimator = cv_results['estimator'][cv_results['test_score'].argmax()]
        logging.info("Performed cross validation")
        return best_estimator
    
    except Exception as e:
        raise CustomException(e,sys)
    
# evaluate the model    
def evaluate_models(X_train,y_train,X_test,y_test,models,params):
    try:
        
        report = {}
        
        for i in range(len(list(models))):
            
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]
            
            rs = GridSearchCV(model,param,cv=3)
            rs.fit(X_train, y_train)
            
            model.set_params(**rs.best_params_)
            model.fit(X_train,y_train)
            
            # model.fit(X_train,y_train)
            
            y_test_pred = model.predict(X_test)
            
            test_model_score = roc_auc_score(y_test, y_test_pred)
            
            report[list(models.keys())[i]] = test_model_score
            
        return report
    
    except Exception as e:
        raise CustomException(e,sys)

def get_metrics(y_pred, y_true):
    try:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        roc_score = roc_auc_score(y_true, y_pred)
        
        return {'Accuracy':accuracy, 'Precision':precision, 
                'Recall':recall, 'F1_Score':f1,
                'ROC_AUC_Score':roc_score}
    
    except Exception as e:
        raise CustomException(e, sys)
    

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise CustomException(e,sys)
    
# def confustion_matrix(y_test, y_test_pred,file_path):
#     cm = confusion_matrix(y_test, y_test_pred)
#     classes = np.unique(y_test)
#     fig, ax = plt.subplots()
#     im = ax.imshow(cm, cmap='Blues')
#     ax.set_xticks(np.arange(len(classes)))
#     ax.set_yticks(np.arange(len(classes)))
#     ax.set_xticklabels(classes)
#     ax.set_yticklabels(classes)
#     ax.set_ylabel('True label')
#     ax.set_xlabel('Predicted label')
#     for i in range(len(classes)):
#         for j in range(len(classes)):
#             text = ax.text(j,i.cm[i,j],ha="center", va="center", color="white")
#     cbar = ax.figure.colorbar(im, ax=ax)
#     plt.savefig(file_path, dpi=300, bbox_inches='tight')
    
# def save_roc_auc_curve(y_test, y_score, file_path):
#     fpr, tpr, thresholds = roc_curve(y_test, y_score)
#     roc_auc = roc_auc_score(y_test, y_score)
#     plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
#     plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver operating characteristic')
#     plt.legend(loc="lower right")
#     plt.savefig(file_path, dpi=300, bbox_inches='tight')