import json 
import os 
import sys
from dotenv import load_dotenv
load_dotenv()
project_home_path = os.environ.get('PROJECT_HOME_PATH')
sys.path.insert(0, project_home_path)
from src.utils import *
import pytest 
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "MLOps best practices\n\n Source code: https://www.github.com/karan842/mlops-best-practices"}

def test_predict_custom_data_valid():
    data = {
        "CreditScore": 700,
        "Geography": "France",
        "Gender": "Male",
        "Age": 35,
        "Tenure": 5,
        "Balance": 2500.0,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 50000.0
    }
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert response.json() == {"Churn Prediction": "No"}

def test_predict_custom_data_invalid_input():
    data = {
        "CreditScore": "invalid_value",
        "Geography": "Spain",
        "Gender": "Female",
        "Age": 18,
        "Tenure": 0,
        "Balance": 1000.0,
        "NumOfProducts": 3,
        "HasCrCard": 0,
        "IsActiveMember": 0,
        "EstimatedSalary": 20000.0
    }
    response = client.post("/predict", json=data)
    assert response.status_code == 422  # 422 indicates validation error
