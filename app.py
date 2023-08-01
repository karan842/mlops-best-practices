from fastapi import FastAPI
import uvicorn
import pandas as pd
from pydantic import BaseModel
import sys
import os
from src.pipeline.predict_pipeline import PredictPipeline, CustomData
from src.exception import CustomException
from src.logger import logging
from src.pipeline.validation_pipeline import CustomDataModel


# Iniiate FastAPI
app = FastAPI()
predictor = PredictPipeline()

@app.get("/")
def home():
    logging.info("Recieved a request at / endpoint.")
    return {"message": "MLOps best practices\n\n Source code: https://www.github.com/karan842/mlops-best-practices"}

@app.post("/")
def home():
    logging.error("Wrong method selected.")
    return {"message":"Wrong method selected1 please use GET method."}

@app.post("/predict")
async def predict_custom_data(custom_data: CustomDataModel):
    try:
        # convert the received Pydantic model to dict
        custom_data_dict = custom_data.dict()
        
        # Create a CustomData instance using the data from Pydantic model
        custom_data_instance = CustomData(**custom_data_dict)
        
        # Get the data as a dataframe from the CustomData instance
        custom_data_df = custom_data_instance.get_data_as_data_frame()
        
        # Make predictions using the PredictPipeline
        preds = predictor.predict(custom_data_df)
        
        # Coverting to the desired output format
        prediction_result = int(preds.item())
        
        if prediction_result == 1:
            return {'Churn Prediction': "Yes"}
        else:
            return {"Churn Prediction": "No"}
        logging.info("Prediction successful.")
        # return {"prediction": prediction_result}
        
    except Exception as e:
        logging.error("Something went wrong on /predict endpoint.")
        return {"Error:": str(e)}  

@app.get("/predict")
def predict_custom_data():
    logging.error("Wrong method selected.")
    return {"message":"Wrong method selected! please use POST method."}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=4040)
