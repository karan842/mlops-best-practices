import os
import sys
import requests
import argparse
from dotenv import load_dotenv
load_dotenv()
project_home_path = os.environ.get('PROJECT_HOME_PATH')
sys.path.append(project_home_path)
from src.pipeline.predict_pipeline import CustomData
from src.logger import logging

# Sample test cases
BASE_URL = "http://localhost:4040"

def predict_endpoint(test_data):
    try:
        response = requests.post(f"{BASE_URL}/predict",
                                 json=test_data)
        response_data = response.json()
        
        if response.status_code == 200:
            prediction = response_data.get("Churn Prediction")
            logging.info(f"Prediction result: {prediction}")
            print(f"Prediction result: {prediction}")
        else:
            logging.error(f"Failed to make a prediction. Error: {response_data}")
            print(f"Failed to make a prediction. Error: {response_data}")
            
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to make request to the server. Error: {e}")
        print(f"Failed to make a request to the server. Error: {e}")
        
if __name__ == '__main__':
    ''' 
    Run this script in a terminal for testing
    ''' 
    parser = argparse.ArgumentParser(description="Test FastAPI predict endpoint.")
    parser.add_argument("--CreditScore", type=int, required=True)
    parser.add_argument("--Geography", type=str, required=True)
    parser.add_argument("--Gender", type=str, required=True)
    parser.add_argument("--Age", type=int, required=True)
    parser.add_argument("--Tenure", type=int, required=True)
    parser.add_argument("--Balance", type=float, required=True)
    parser.add_argument("--NumOfProducts", type=int, required=True)
    parser.add_argument("--HasCrCard", type=int, required=True)
    parser.add_argument("--IsActiveMember", type=int, required=True)
    parser.add_argument("--EstimatedSalary", type=float, required=True)
    
    args = parser.parse_args()

    test_data = {
        "CreditScore": args.CreditScore,
        "Geography": args.Geography,
        "Gender": args.Gender,
        "Age": args.Age,
        "Tenure": args.Tenure,
        "Balance": args.Balance,
        "NumOfProducts": args.NumOfProducts,
        "HasCrCard": args.HasCrCard,
        "IsActiveMember": args.IsActiveMember,
        "EstimatedSalary": args.EstimatedSalary
    }
    
    predict_endpoint(test_data)

    