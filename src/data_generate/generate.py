import pandas as pd
from faker import Faker
import random
import os
import sys
from dotenv import load_dotenv
load_dotenv()
project_home_path = os.environ.get('PROJECT_HOME_PATH')
sys.path.append(project_home_path)
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class SyntheticDataGenerationConfig:
    synthetic_data_file_path = os.path.join('synthetic_data','synthetic_data.csv')

class SyntheticDataGenerator:
    
    def __init__(self,data_path,num_rows_to_add):
        self.df = pd.read_csv(data_path)
        self.num_rows_to_add = num_rows_to_add
        self.fake = Faker()
        self.unique_countries = self.df['Geography'].unique().tolist()
        self.gender = self.df['Gender'].unique().tolist()
        self.synthetic_data_config = SyntheticDataGenerationConfig()
        
        
    def generate_synthetic_data(self):
        try:
            
            new_rows = []
            for _ in range(self.num_rows_to_add):
                row = {}
                row['RowNumber'] = self.df['RowNumber'].max() + 1
                row['CustomerId'] = self.fake.uuid4()
                row['Surname'] = self.fake.last_name()
                row['CreditScore'] = self.fake.random_int(min=300,max=900)
                row['Geography'] = self.fake.random.choice(self.unique_countries)  
                row['Age'] = self.fake.random_int(min=18,max=95)          
                row['Gender'] = self.fake.random.choice(self.gender) 
                row['Tenure'] = self.fake.random_int(min=0,max=10)
                row['Balance'] = round(self.fake.random.uniform(0.0,260898.09),2)
                row['NumOfProducts'] = self.fake.random_int(min=1,max=4)
                row['HasCrCard'] = self.fake.random_int(min=0,max=1)
                row['IsActiveMember'] = self.fake.random_int(min=0,max=1)
                row['EstimatedSalary'] = round(self.fake.random.uniform(10.58, 199998.88), 2)
                row['Exited'] = self.fake.random_int(min=0,max=1)
                new_rows.append(row)
            new_df = pd.DataFrame(new_rows)
            synthetic_data = pd.concat([self.df,new_df],ignore_index=True)
            
            os.makedirs(os.path.dirname(self.synthetic_data_config.synthetic_data_file_path))
            synthetic_data.to_csv(self.synthetic_data_config.synthetic_data_file_path,index=False,header=True)
        
            return(
                self.synthetic_data_config.synthetic_data_file_path,
            )
        
        except Exception as e:
            raise CustomException(e,sys)
    
    
if __name__ == "__main__":       
    # gen_df = SyntheticDataGenerator('raw_data\data.csv',10000)
    # synthetic_data = gen_df.generate_synthetic_data()
    # logging.info("Synthetic data generated")
    print('Working')