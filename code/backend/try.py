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