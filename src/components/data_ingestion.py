import sys
import os
from dotenv import load_dotenv
load_dotenv()
project_home_path = os.environ.get('PROJECT_HOME_PATH')
sys.path.append(project_home_path)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.tracking.params import save_params_to_yaml
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str= os.path.join('artifacts',"train.csv")
    test_data_path: str= os.path.join('artifacts',"test.csv")
    raw_data_path: str= os.path.join('artifacts',"raw.csv")
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv('synthetic_data\synthetic_data.csv')
            df.drop(['RowNumber','Surname','CustomerId'],axis=1,inplace=True)
            logging.info("Read the dataset as dataframe")
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("Train test split inititated")
            train_set,test_set=train_test_split(df,test_size=0.3,random_state=42)
            
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            
            logging.info("Ingestion of the data is completed!")
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

            
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    print('\n\n------Dev Stage Training------\n\n')
    print("> Data Ingestion process begins....")
    obj = DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()
    print('> Data Ingestion process completed.\n')
    
    print('> Data Transformation operation begings....')
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data,test_data)
    print('> Data Trasformation operation completed.\n')
    # print(X.shape,y.shape)
    
    print('> Training the machine learning algorithm (include experiment tracking and tuning)....\n')
    save_params_to_yaml()
    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr,test_arr))
    print('\n> Model training and evaluation process done.\n')