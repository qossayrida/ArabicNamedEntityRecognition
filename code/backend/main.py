import sys
import io
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from backend import assemblage
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from joblib import load
import numpy as np
import warnings
import json


# Change the standard output encoding to utf-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')
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


# Load the saved CRF model
crf_loaded = load('model/crf_model.joblib')

# Load the saved model, tokenizer, label encoder, and configuration
model = load_model('model/ner_model.keras')
tokenizer = load('model/tokenizer.joblib')
label_encoder = load('model/label_encoder.joblib')
with open('model/config.json', 'r') as config_file:
    config = json.load(config_file)
max_length = config['max_length']

@app.post("/predict")
def predict(request: NERRequest):
    text = request.text
    model = request.model

    if model == "CRF":
        new_tokens = text.split()
        new_features = assemblage.extract_features(new_tokens)
        predictions = crf_loaded.predict([new_features])
        entities = [
            {"entity": label, "value": token}
            for token, label in zip(new_tokens, predictions[0])
        ]
    elif model == "RNN":
        result = predict_sentence(text)
        entities = [
            {"entity": label, "value": word}
            for word, label in result
        ]
    elif model == "Transformer":
        entities = [{"entity": "LOC", "value": "المجدل", "start": 65, "end": 72}]
    else:
        entities = [{"entity": "MON", "value": "500", "start": 15, "end": 18}]

    return {"entities": entities}


def predict_sentence(sentence):

    # Tokenize and pad the sentence
    sequence = tokenizer.texts_to_sequences([sentence])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')

    # Predict the labels
    predictions = model.predict(padded_sequence)
    predicted_labels = [label_encoder.inverse_transform([np.argmax(p)])[0] for p in predictions[0]]

    # Return the words and their predicted labels
    return list(zip(sentence.split(), predicted_labels))