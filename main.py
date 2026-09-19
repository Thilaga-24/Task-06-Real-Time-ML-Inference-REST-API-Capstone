# ============================================================
# REAL-TIME ML INFERENCE REST API - COMPLETE SINGLE FILE
# ============================================================

import os
import joblib

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# ============================================================
# 1. CREATE / LOAD MODEL
# ============================================================

MODEL_PATH = "sentiment_model.joblib"


def create_model():

    # Sample training data
    # Replace this with your actual dataset if required

    texts = [
        "I love this product",
        "This product is amazing",
        "Excellent service",
        "Very happy with the purchase",
        "I really enjoyed this",
        "The product is fantastic",
        "This is wonderful",
        "Absolutely great product",

        "I hate this product",
        "This product is terrible",
        "Very bad service",
        "I am disappointed",
        "Worst product ever",
        "I don't like this",
        "This is horrible",
        "Very poor product"
    ]

    labels = [
        1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0
    ]

    # ML pipeline
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                stop_words="english"
            )
        ),

        (
            "classifier",
            LogisticRegression(
                max_iter=1000
            )
        )
    ])

    # Train
    model.fit(texts, labels)

    # Save
    joblib.dump(model, MODEL_PATH)

    print("Model trained and saved successfully.")


# ============================================================
# 2. LOAD MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):
    create_model()

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# 3. CREATE FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Real-Time ML Inference API",
    description="Sentiment Classification REST API",
    version="1.0.0"
)


# ============================================================
# 4. REQUEST SCHEMA
# ============================================================

class PredictionRequest(BaseModel):

    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Text that needs sentiment classification"
    )


# ============================================================
# 5. RESPONSE SCHEMA
# ============================================================

class PredictionResponse(BaseModel):

    prediction: int

    sentiment: str

    confidence: float

    probabilities: dict[str, float]


# ============================================================
# 6. ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Real-Time ML Inference API is running",
        "version": "1.0.0",
        "documentation": "/docs"
    }


# ============================================================
# 7. HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True
    }


# ============================================================
# 8. PREDICTION ENDPOINT
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(request: PredictionRequest):

    try:

        # Get prediction
        prediction = model.predict(
            [request.text]
        )[0]

        # Get probabilities
        probabilities = model.predict_proba(
            [request.text]
        )[0]

        # Model classes
        classes = model.classes_

        probability_dict = {
            str(label): float(probability)
            for label, probability in zip(
                classes,
                probabilities
            )
        }

        # Convert prediction to sentiment
        if prediction == 1:
            sentiment = "positive"
        else:
            sentiment = "negative"

        # Confidence
        confidence = float(
            max(probabilities)
        )

        return {
            "prediction": int(prediction),
            "sentiment": sentiment,
            "confidence": confidence,
            "probabilities": probability_dict
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# 9. RUN SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )