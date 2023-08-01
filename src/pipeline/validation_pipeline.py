from typing import List, Dict
from pydantic import BaseModel, validator, ValidationError

# Define the allowed_values dictionary directly in the script
allowed_values: Dict[str, Dict[str, List[str]]] = {
    "CreditScore": {
        "min": 300,
        "max": 850
    },
    "Geography": ["France", "Germany", "Spain"],
    "Gender": ["Male", "Female"],
    "Age": {
        "min": 18,
        "max": 100
    },
    "Tenure": {
        "min": 0,
        "max": 10
    },
    "Balance": {
        "min": 0.0,
        "max": 1000000.0
    },
    "NumOfProducts": {
        "min": 1,
        "max": 4
    },
    "HasCrCard": {
        "min": 0,
        "max": 1
    },
    "IsActiveMember": {
        "min": 0,
        "max": 1
    },
    "EstimatedSalary": {
        "min": 0.0,
        "max": 2000000.0
    }
}

# Define the CustomDataModel class using the allowed_values dictionary
class CustomDataModel(BaseModel):
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

    @validator('CreditScore')
    def validate_credit_score(cls, value):
        if not allowed_values['CreditScore']['min'] <= value <= allowed_values['CreditScore']['max']:
            raise ValueError(f"CreditScore must be between {allowed_values['CreditScore']['min']} and {allowed_values['CreditScore']['max']}")
        return value

    @validator('Geography')
    def validate_geography(cls, value):
        if value not in allowed_values['Geography']:
            raise ValueError(f"Geography must be one of {', '.join(allowed_values['Geography'])}")
        return value

    @validator('Gender')
    def validate_gender(cls, value):
        if value not in allowed_values['Gender']:
            raise ValueError(f"Gender must be one of {', '.join(allowed_values['Gender'])}")
        return value

    @validator('Age')
    def validate_age(cls, value):
        if not allowed_values['Age']['min'] <= value <= allowed_values['Age']['max']:
            raise ValueError(f"Age must be between {allowed_values['Age']['min']} and {allowed_values['Age']['max']}")
        return value

    @validator('Tenure')
    def validate_tenure(cls, value):
        if not allowed_values['Tenure']['min'] <= value <= allowed_values['Tenure']['max']:
            raise ValueError(f"Tenure must be between {allowed_values['Tenure']['min']} and {allowed_values['Tenure']['max']}")
        return value

    @validator('Balance')
    def validate_balance(cls, value):
        if not allowed_values['Balance']['min'] <= value <= allowed_values['Balance']['max']:
            raise ValueError(f"Balance must be between {allowed_values['Balance']['min']} and {allowed_values['Balance']['max']}")
        return value

    @validator('NumOfProducts')
    def validate_num_of_products(cls, value):
        if not allowed_values['NumOfProducts']['min'] <= value <= allowed_values['NumOfProducts']['max']:
            raise ValueError(f"NumOfProducts must be between {allowed_values['NumOfProducts']['min']} and {allowed_values['NumOfProducts']['max']}")
        return value

    @validator('HasCrCard')
    def validate_has_cr_card(cls, value):
        if not allowed_values['HasCrCard']['min'] <= value <= allowed_values['HasCrCard']['max']:
            raise ValueError(f"HasCrCard must be between {allowed_values['HasCrCard']['min']} and {allowed_values['HasCrCard']['max']}")
        return value

    @validator('IsActiveMember')
    def validate_is_active_member(cls, value):
        if not allowed_values['IsActiveMember']['min'] <= value <= allowed_values['IsActiveMember']['max']:
            raise ValueError(f"IsActiveMember must be between {allowed_values['IsActiveMember']['min']} and {allowed_values['IsActiveMember']['max']}")
        return value

    @validator('EstimatedSalary')
    def validate_estimated_salary(cls, value):
        if not allowed_values['EstimatedSalary']['min'] <= value <= allowed_values['EstimatedSalary']['max']:
            raise ValueError(f"EstimatedSalary must be between {allowed_values['EstimatedSalary']['min']} and {allowed_values['EstimatedSalary']['max']}")
        return value

