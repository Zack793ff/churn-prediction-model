from fastapi import FastAPI, HTTPException
from enum import Enum
from pydantic import BaseModel, Field, conint
import pickle
import uvicorn

from app.churn_serving import predict_single

# Create FastAPI app instance
app = FastAPI()

# Function to predict churn for a single customer
def predict_single(customer, dv, model):
    X = dv.transform([customer])  # Transform the customer data
    y_pred = model.predict_proba(X)[:, 1]  # Predict probability of churn
    return y_pred[0]  # Return the churn probability

# Load the model and DictVectorizer at startup
with open('churn-model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

# Enums for binary fields
class YesNoEnum(str, Enum):
    yes = 'yes'
    no = 'no'

class InternetServiceEnum(str, Enum):
    dsl = 'dsl'
    fiber_optic = 'fiber_optic'
    no = 'no'

class ContractEnum(str, Enum):
    month_to_month = 'month-to-month'
    one_year = 'one_year'
    two_year = 'two_year'

class PaymentMethodEnum(str, Enum):
    electronic_check = 'electronic_check'
    mailed_check = 'mailed_check'
    bank_transfer_automatic = 'bank_transfer_(automatic)'
    credit_card_automatic = 'credit_card_(automatic)'

# Define a Pydantic model to handle the request body
class Customer(BaseModel):
    customerid: str = Field(..., description="Unique customer ID")
    gender: str = Field(..., description="Customer's gender: 'female' or 'male'", pattern="^(male|female)$")
    seniorcitizen: conint(ge=0, le=1) = Field(..., description="Whether the customer is a senior citizen (0 or 1)")
    partner: YesNoEnum = Field(..., description="Does the customer have a partner? 'yes' or 'no'")
    dependents: YesNoEnum = Field(..., description="Does the customer have dependents? 'yes' or 'no'")
    tenure: conint(ge=0) = Field(..., description="Number of months the customer has been with the company")
    phoneservice: YesNoEnum = Field(..., description="Does the customer have phone service? 'yes' or 'no'")
    multiplelines: YesNoEnum = Field(..., description="Does the customer have multiple lines? 'yes' or 'no'")
    internetservice: InternetServiceEnum = Field(..., description="Customer's internet service: 'dsl', 'fiber_optic', or 'no'")
    onlinesecurity: YesNoEnum = Field(..., description="Does the customer have online security? 'yes' or 'no'")
    onlinebackup: YesNoEnum = Field(..., description="Does the customer have online backup? 'yes' or 'no'")
    deviceprotection: YesNoEnum = Field(..., description="Does the customer have device protection? 'yes' or 'no'")
    techsupport: YesNoEnum = Field(..., description="Does the customer have tech support? 'yes' or 'no'")
    streamingtv: YesNoEnum = Field(..., description="Does the customer stream TV? 'yes' or 'no'")
    streamingmovies: YesNoEnum = Field(..., description="Does the customer stream movies? 'yes' or 'no'")
    contract: ContractEnum = Field(..., description="Customer's contract type: 'month-to-month', 'one_year', or 'two_year'")
    paperlessbilling: YesNoEnum = Field(..., description="Does the customer use paperless billing? 'yes' or 'no'")
    paymentmethod: PaymentMethodEnum = Field(..., description="Customer's payment method")
    monthlycharges: float = Field(..., description="Customer's monthly charges")
    totalcharges: float = Field(..., description="Customer's total charges")

# Define a POST route for /predict
@app.post('/predict')
def predict(customer: Customer):
    try:
        customer_data = customer.dict()  # Use dict for Pydantic models
        prediction = predict_single(customer_data, dv, model)  # Pass customer data to predict
        churn = prediction >= 0.5

        result = {
            'churn_probability': float(prediction),
            'churn': bool(churn),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return result

# Run the FastAPI app using uvicorn
if __name__ == '__main__':
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
