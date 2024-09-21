from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('sentiment_model.pkl')

# Initialise FastAPI
app = FastAPI()

# Define the input data structure
class Review(BaseModel):
    review_headline: str
    review_body: str

# Define a prediction endpoint
@app.post("/predict/")
def predict(review: Review):
    # Combine the input for prediction (you may need to adjust this part based on your preprocessing)
    input_data = pd.DataFrame([[review.review_headline + ' ' + review.review_body]], columns=['text'])
    
    # Make a prediction
    sentiment = model.predict(input_data)[0]
    
    # Map numerical sentiment to labels (assuming 0: negative, 1: neutral, 2: positive)
    sentiment_label = {0: "negative", 1: "neutral", 2: "positive"}
    
    return {"sentiment": sentiment_label[sentiment]}
