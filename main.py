# Import required libraries for FastAPI deployment
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import os

# 1. Initialize FastAPI Application
app = FastAPI(
    title="Walmart Retail Analytics API System",
    description="Institutional-grade backend API powered by an optimized XGBoost Regressor (97.79% Accuracy) for forecasting weekly sales.",
    version="1.0.0"
)

# 2. Define the path and securely load the trained production model from artifacts folder
MODEL_PATH = "artifacts/walmart_xgboost_model.pkl"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    raise FileNotFoundError(f"Production model not found at the specified location: {MODEL_PATH}")

# 3. Define the Input Data Schema using Pydantic for automated request validation
class SalesInputSchema(BaseModel):
    Store: int
    Holiday_Flag: int
    Temperature: float
    Fuel_Price: float
    CPI: float
    Unemployment: float
    Year: int
    Month: int
    Week_Of_Year: int

    # This creates a default placeholder example inside Swagger UI
    class Config:
        json_schema_extra = {
            "example": {
                "Store": 1,
                "Holiday_Flag": 0,
                "Temperature": 42.31,
                "Fuel_Price": 2.572,
                "CPI": 211.0963582,
                "Unemployment": 8.106,
                "Year": 2010,
                "Month": 2,
                "Week_Of_Year": 5
            }
        }

# 4. Create the Root Endpoint (Welcome Page)
@app.get("/", tags=["General"])
def read_root():
    return {
        "Status": "Online",
        "System_Name": "Walmart Predictive Sales Forecasting API Engine",
        "Documentation_URL": "/docs"
    }

# 5. Create the Core Prediction Endpoint (POST Route)
@app.post("/predict", tags=["Forecasting Engine"])
def forecast_weekly_sales(inputs: SalesInputSchema):
    try:
        # Convert JSON input fields into the strict array format required by XGBoost
        structured_features = np.array([[
            inputs.Store,
            inputs.Holiday_Flag,
            inputs.Temperature,
            inputs.Fuel_Price,
            inputs.CPI,
            inputs.Unemployment,
            inputs.Year,
            inputs.Month,
            inputs.Week_Of_Year
        ]])

        # Execute prediction using the loaded model
        prediction_output = model.predict(structured_features)

        # Return the final forecast result as a JSON response
        return {
            "Success": True,
            "Store_ID": inputs.Store,
            "Target_Year": inputs.Year,
            "Target_Week": inputs.Week_Of_Year,
            "Predicted_Weekly_Sales_USD": float(prediction_output[0])
        }

    except Exception as error_details:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during backend model processing: {str(error_details)}"
        )
