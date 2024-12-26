import sys
import io
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from joblib import load
from backend import Collection

# Change the standard output encoding to utf-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change for production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

class NERRequest(BaseModel):
    text: str
    model: str


@app.post("/predict")
def predict(request: NERRequest):
    text = request.text
    model = request.model

    if model == "CRF":
        crf_loaded = load('model/crf_model.joblib')
        new_tokens = text.split()
        new_features = Collection.extract_features(new_tokens)
        predictions = crf_loaded.predict([new_features])
        entities = [
            {"entity": label, "value": token}
            for token, label in zip(new_tokens, predictions[0])
        ]
    elif model == "RNN":
        entities = [{"entity": "PER", "value": "عثمان زقوت", "start": 50, "end": 62}]
    elif model == "Transformer":
        entities = [{"entity": "LOC", "value": "المجدل", "start": 65, "end": 72}]
    else:
        entities = [{"entity": "MON", "value": "500", "start": 15, "end": 18}]

    return {"entities": entities}