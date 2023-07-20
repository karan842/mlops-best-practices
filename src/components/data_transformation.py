import sys
import os
from dotenv import load_dotenv
load_dotenv()
project_home_path = os.environ.get('PROJECT_HOME_PATH')
sys.path.append(project_home_path)
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, OrdinalEncoder, LabelEncoder

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, fix_outliers, class_imbalance, handling_class_imbalance

from configure import configure


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
        
    def get_data_transformer_object(self):
        
        '''
        This function is transforming the data
        '''
        
        try:
            numerical_columns = configure["data_transformation"]["numerical"]["columns"]
            categorical_columns = configure["data_transformation"]["categorical"]["columns"]
            
            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="mean")),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )
            
            categorical_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )
            
            logging.info('Numerical columns scaling completed!')
            logging.info("Categorical columns encoding completed!")
            
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",numerical_pipeline,numerical_columns),
                    ("cat_pipeline",categorical_pipeline,categorical_columns)
                ]
            )
            
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
        
    def initiate_data_transformation(self,train_path,test_path):
        
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Loading The train and test data completed!")
            logging.info("Obtaining preprocessing object")
            
            preprocessing_obj = self.get_data_transformer_object()
            
            target_column_name = configure["data_transformation"]["target_column"]
            numerical_columns = configure["data_transformation"]["numerical"]["columns"]
            
            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            input_feature_train_df.applymap(fix_outliers)
            target_feature_train_df = train_df[target_column_name]
                
            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info("Applying preprocessing object on test dataframes and testing dataframes.")
            
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            target_feature_train_arr = np.array(target_feature_train_df)
            target_feature_test_arr = np.array(target_feature_test_df)
            
            imbalance_thresh = configure["data_transformation"]["imbalance_threshold"]
            
            input_feature_train_arr_balanced, target_feature_train_arr_balanced = handling_class_imbalance(input_feature_test_arr,target_feature_test_arr,imbalance_thresh)
            
            train_arr = np.c_[
                input_feature_train_arr_balanced, target_feature_train_arr_balanced
            ]
            
            test_arr = np.c_[
                input_feature_test_arr, target_feature_test_arr
            ]
              
                
            logging.info("Saving preprocessing object.")
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
                 
        except Exception as e:
            raise CustomException(e,sys)