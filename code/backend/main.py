import sys
import io
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from backend import assemblage
from tensorflow.keras.preprocessing.sequence import pad_sequences
from joblib import load
import numpy as np
import warnings
import tensorflow as tf
import pickle


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

# Load the model and make predictions
max_len = 100
rnn_model = tf.keras.models.load_model("model/arabic_ner_model.h5")
with open("model/word2idx.pkl", "rb") as file:
    word2idx = pickle.load(file)
with open("model/idx2label.pkl", "rb") as file:
    idx2label = pickle.load(file)

@app.post("/predict")
def predict(request: NERRequest):
    text = request.text
    model = request.model

    if model == "CRF":
        new_tokens = text.split()
        new_features = assemblage.extract_features_for_crf(new_tokens)
        predictions = crf_loaded.predict([new_features])
        entities = [
            {"entity": label, "value": token}
            for token, label in zip(new_tokens, predictions[0])
        ]
    elif model == "RNN":
        sample_sentence = text.split()
        sample_indices = [word2idx.get(word, word2idx["<UNK>"]) for word in sample_sentence]
        sample_padded = pad_sequences([sample_indices], maxlen=max_len, padding="post")
        predictions = rnn_model.predict(sample_padded)
        predicted_tags = [idx2label[np.argmax(tag)] for tag in predictions[0]]
        entities = [
            {"entity": label, "value": token}
            for token, label in zip(sample_sentence, predicted_tags)
        ]
    elif model == "DT":
        entities = [{"entity": "LOC", "value": "المجدل", "start": 65, "end": 72}]
    elif model == "NB":
        entities = [{"entity": "LOC", "value": "المجدل", "start": 65, "end": 72}]
    else:
        entities = [{"entity": "MON", "value": "500", "start": 15, "end": 18}]

    return {"entities": entities}


