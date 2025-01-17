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

# Load the saved model
try:
    model = joblib.load("app/trained_model.pkl")
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading the model: {e}")
    raise

# Define the app
app = FastAPI()

# Define a request body schema
# to do: some variables are in fact int - update
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

# Endpoint for prediction
@app.post("/predict/")
def predict(request: PredictionRequest):
    # Log the incoming request
    logger.info(f"Received request: {request.dict()}")

    # Convert request data to DataFrame
    input_df = pd.DataFrame([request.dict()])
    logger.debug(f"Input DataFrame: {input_df}")

    # Make a prediction
    try:
        prediction = model.predict(input_df)
        logger.info(f"Prediction successful: {prediction[0]}")
        return {"prediction": float(prediction[0])}
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return {"error": "Failed to make a prediction"}