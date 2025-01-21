from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Format logs
)
logger = logging.getLogger(__name__)  # Get a logger for the app

# Load the saved pipeline and model
try:
    pipeline = joblib.load("app/pipeline.pkl")  # Load the preprocessing pipeline
    model = joblib.load("app/trained_model.pkl")  # Load the trained model
    logger.info("Pipeline and model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading the pipeline or model: {e}")
    raise

# Define the app
app = FastAPI()

# Define a request body schema
# Adjust types if some variables should be integers
class PredictionRequest(BaseModel):
    bar: float
    baz: float
    xgt: float
    qgg: float
    lux: float
    wsg: float
    yyz: float
    drt: float
    gox: float
    foo: float
    boz: float
    fyt: float
    lgh: float
    hrt: float
    juu: float
    date: str

# Endpoint for prediction
@app.post("/predict/")
def predict(request: PredictionRequest):
    # selected features
    numerical_features = ['drt', 'bar', 'xgt', 'qgg', 'lux', 'yyz', 'gox', 'foo', 'hrt', 'juu']
    cat_features = ['fyt', 'lgh', 'year']
    selected_features = numerical_features + cat_features
    # Log the incoming request
    logger.info(f"Received request: {request.dict()}")

    # Convert request data to DataFrame
    input_df = pd.DataFrame([request.dict()])

    #parse date feature
    input_df["date"] = pd.to_datetime(input_df["date"])
    input_df["year"] = input_df["date"].dt.year

    logger.debug(f"Input DataFrame: {input_df}")

    # Preprocess the input data with the pipeline and make a prediction
    try:
        matrix_pp = pipeline.transform(input_df[numerical_features]) # Preprocess data using pipeline
        input_df_pp = pd.DataFrame(matrix_pp, columns=numerical_features ) #turn preprocess data into dataframe
        input_df_pp[cat_features] = input_df[cat_features] #complete missing features
        logger.debug(f"Processed Input DataFrame: {input_df_pp}")

        prediction = model.predict(input_df_pp[selected_features])  # Predict using the model
        logger.info(f"Prediction successful: {prediction[0]}")

        return {"prediction": float(prediction[0])}
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return {"error": "Failed to make a prediction"}
