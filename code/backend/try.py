import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from joblib import load
import numpy as np
import json

# Load the saved model, tokenizer, label encoder, and configuration
model = load_model('../model/ner_model.keras')
tokenizer = load('../model/tokenizer.joblib')
label_encoder = load('../model/label_encoder.joblib')

# Load max_length
with open('../model/config.json', 'r') as config_file:
    config = json.load(config_file)
max_length = config['max_length']


# Function to predict labels for a sentence
def predict_sentence(sentence):
    # Tokenize and pad the sentence
    sequence = tokenizer.texts_to_sequences([sentence])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')

    # Predict the labels
    predictions = model.predict(padded_sequence)
    predicted_labels = [label_encoder.inverse_transform([np.argmax(p)])[0] for p in predictions[0]]

    # Return the words and their predicted labels
    return list(zip(sentence.split(), predicted_labels))


# Test the function with a new sentence
test_sentence = "صورة لعملة ورقية من فئة 500 ملز خلال فترة الانتداب البريطاني على فلسطين."

result = predict_sentence(test_sentence)

# Print the results
print("Input Sentence:")
print(test_sentence)
print("\nPredicted Labels:")
for word, label in result:
    print(f"{word}: {label}")
