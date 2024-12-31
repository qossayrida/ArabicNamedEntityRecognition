# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from joblib import load
# import numpy as np
# import json
#
# # Load the saved model, tokenizer, label encoder, and configuration
# model = load_model('../model/ner_model.keras')
# tokenizer = load('../model/tokenizer.joblib')
# label_encoder = load('../model/label_encoder.joblib')
#
# # Load max_length
# with open('../model/config.json', 'r') as config_file:
#     config = json.load(config_file)
# max_length = config['max_length']
#
#
# # Function to predict labels for a sentence
# def predict_sentence(sentence):
#     # Tokenize and pad the sentence
#     sequence = tokenizer.texts_to_sequences([sentence])
#     padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
#
#     # Predict the labels
#     predictions = model.predict(padded_sequence)
#     predicted_labels = [label_encoder.inverse_transform([np.argmax(p)])[0] for p in predictions[0]]
#
#     # Return the words and their predicted labels
#     return list(zip(sentence.split(), predicted_labels))
#
#
# # Test the function with a new sentence
# test_sentence = "صورة لعملة ورقية من فئة 500 ملز خلال فترة الانتداب البريطاني على فلسطين."
#
# result = predict_sentence(test_sentence)
#
# # Print the results
# print("Input Sentence:")
# print(test_sentence)
# print("\nPredicted Labels:")
# for word, label in result:
#     print(f"{word}: {label}")

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

max_len = 100
# Load the model and make predictions
loaded_model = tf.keras.models.load_model("../model/arabic_ner_model.h5")

# Load mappings
with open("../model/word2idx.pkl", "rb") as file:
    word2idx = pickle.load(file)
with open("../model/idx2label.pkl", "rb") as file:
    idx2label = pickle.load(file)

# Prepare a sample sentence for prediction
sample_sentence = "صورة لعملة ورقية من فئة 500 ملز خلال فترة الانتداب البريطاني على فلسطين.".split()
sample_indices = [word2idx.get(word, word2idx["<UNK>"]) for word in sample_sentence]
sample_padded = pad_sequences([sample_indices], maxlen=max_len, padding="post")
predictions = loaded_model.predict(sample_padded)
predicted_tags = [idx2label[np.argmax(tag)] for tag in predictions[0]]

# Print predictions
print("Sentence:", sample_sentence)
print("Predicted Tags:", predicted_tags[:len(sample_sentence)])


# Prepare a sample sentence for prediction
sample_sentence = "انا اسمي قصي ابو ريدة من قرية قصرة ادرس علم حاسوب في جامعة بيرزيت".split()
sample_indices = [word2idx.get(word, word2idx["<UNK>"]) for word in sample_sentence]
sample_padded = pad_sequences([sample_indices], maxlen=max_len, padding="post")
predictions = loaded_model.predict(sample_padded)
predicted_tags = [idx2label[np.argmax(tag)] for tag in predictions[0]]

# Print predictions
print("Sentence:", sample_sentence)
print("Predicted Tags:", predicted_tags[:len(sample_sentence)])

# Prepare a sample sentence for prediction
sample_sentence = "مرحبا انا اسمي قصي عبد موسى واسكن في قرية قصرة".split()
sample_indices = [word2idx.get(word, word2idx["<UNK>"]) for word in sample_sentence]
sample_padded = pad_sequences([sample_indices], maxlen=max_len, padding="post")
predictions = loaded_model.predict(sample_padded)
predicted_tags = [idx2label[np.argmax(tag)] for tag in predictions[0]]

# Print predictions
print("Sentence:", sample_sentence)
print("Predicted Tags:", predicted_tags[:len(sample_sentence)])