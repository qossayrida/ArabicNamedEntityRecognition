# import tensorflow as tf
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# import pickle
# import numpy as np
#
# max_len = 100
# # Load the model and make predictions
# loaded_model = tf.keras.models.load_model("../model/arabic_ner_model.h5")
#
# # Load mappings
# with open("../model/word2idx.pkl", "rb") as file:
#     word2idx = pickle.load(file)
# with open("../model/idx2label.pkl", "rb") as file:
#     idx2label = pickle.load(file)
#
# # Prepare a sample sentence for prediction
# sample_sentence = "صورة لعملة ورقية من فئة 500 ملز خلال فترة الانتداب البريطاني على فلسطين.".split()
# sample_indices = [word2idx.get(word, word2idx["<UNK>"]) for word in sample_sentence]
# sample_padded = pad_sequences([sample_indices], maxlen=max_len, padding="post")
# predictions = loaded_model.predict(sample_padded)
# predicted_tags = [idx2label[np.argmax(tag)] for tag in predictions[0]]
#
# # Print predictions
# print("Sentence:", sample_sentence)
# print("Predicted Tags:", predicted_tags[:len(sample_sentence)])
#
#
# # Prepare a sample sentence for prediction
# sample_sentence = "انا اسمي قصي ابو ريدة من قرية قصرة ادرس علم حاسوب في جامعة بيرزيت".split()
# sample_indices = [word2idx.get(word, word2idx["<UNK>"]) for word in sample_sentence]
# sample_padded = pad_sequences([sample_indices], maxlen=max_len, padding="post")
# predictions = loaded_model.predict(sample_padded)
# predicted_tags = [idx2label[np.argmax(tag)] for tag in predictions[0]]
#
# # Print predictions
# print("Sentence:", sample_sentence)
# print("Predicted Tags:", predicted_tags[:len(sample_sentence)])
#
# # Prepare a sample sentence for prediction
# sample_sentence = "مرحبا انا اسمي قصي عبد موسى واسكن في قرية قصرة".split()
# sample_indices = [word2idx.get(word, word2idx["<UNK>"]) for word in sample_sentence]
# sample_padded = pad_sequences([sample_indices], maxlen=max_len, padding="post")
# predictions = loaded_model.predict(sample_padded)
# predicted_tags = [idx2label[np.argmax(tag)] for tag in predictions[0]]
#
# # Print predictions
# print("Sentence:", sample_sentence)
# print("Predicted Tags:", predicted_tags[:len(sample_sentence)])

import keras
import numpy as np
from keras import ops
import tensorflow as tf
from keras import layers
from collections import Counter
from camel_tools.tokenizers.word import simple_word_tokenize
import camel_tools
from tensorflow.keras.optimizers import Adam
import pickle
from keras.saving import register_keras_serializable

from camel_tools.tokenizers.word import simple_word_tokenize


@register_keras_serializable()
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.ffn = keras.Sequential(
            [
                keras.layers.Dense(ff_dim, activation="relu"),
                keras.layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

@register_keras_serializable()
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = keras.layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.pos_emb = keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, inputs):
        maxlen = ops.shape(inputs)[-1]
        positions = ops.arange(start=0, stop=maxlen, step=1)
        position_embeddings = self.pos_emb(positions)
        token_embeddings = self.token_emb(inputs)
        return token_embeddings + position_embeddings
#%% md
### Build the NER model class as a keras.Model subclass

@register_keras_serializable()
class NERModel(keras.Model):
    def __init__(
        self, num_tags, vocab_size, maxlen=128, embed_dim=32, num_heads=6, ff_dim=32, **kwargs
    ):
        super().__init__(**kwargs)
        self.num_tags = num_tags
        self.vocab_size = vocab_size
        self.maxlen = maxlen
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim

        self.embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
        self.transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
        self.dropout1 = layers.Dropout(0.1)
        self.ff = layers.Dense(ff_dim, activation="relu")
        self.dropout2 = layers.Dropout(0.1)
        self.ff_final = layers.Dense(num_tags, activation="softmax")

    def call(self, inputs, training=False):
        x = self.embedding_layer(inputs)
        x = self.transformer_block(x)
        x = self.dropout1(x, training=training)
        x = self.ff(x)
        x = self.dropout2(x, training=training)
        x = self.ff_final(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_tags": self.num_tags,
            "vocab_size": self.vocab_size,
            "maxlen": self.maxlen,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(
            num_tags=config.get("num_tags", 10),  # Default to 10 if missing
            vocab_size=config.get("vocab_size", 5000),  # Default to 5000 if missing
            maxlen=config.get("maxlen", 128),
            embed_dim=config.get("embed_dim", 32),
            num_heads=config.get("num_heads", 6),
            ff_dim=config.get("ff_dim", 32),
            **config
        )


def make_tag_lookup_table():
    iob_labels = ["B", "I"]
    ner_labels = ["PER", "ORG", "LOC", "EVE", 'NUM', 'MON', 'LAN', 'TIME']
    all_labels = [(label1, label2) for label2 in ner_labels for label1 in iob_labels]
    all_labels = ["-".join([a, b]) for a, b in all_labels]
    all_labels = ["[PAD]", "O"] + all_labels
    return dict(zip(range(0, len(all_labels) + 1), all_labels))

mapping = make_tag_lookup_table()


import pickle
import tensorflow as tf
import numpy as np

loaded_model = tf.keras.models.load_model(
    '../model/ner_model_trans.keras',
    custom_objects={
        "NERModel": NERModel,
        "TransformerBlock": TransformerBlock,
        "TokenAndPositionEmbedding": TokenAndPositionEmbedding
    }
)

# Load the lookup layer
with open("../model/lookup_layer.joblib", "rb") as file:
    loaded_lookup_layer = pickle.load(file)

# Test the model with new input
def tokenize_and_convert_to_ids(text, lookup_layer):
    tokens = simple_word_tokenize(text)
    return lookup_layer(tokens)

# Sample input
sample_text = "مرحبا انا اسمي عبد الله وصديقي اسمه محمد"
sample_input = tokenize_and_convert_to_ids(sample_text, loaded_lookup_layer)
sample_input = tf.reshape(sample_input, shape=[1, -1])

# Predict
output = loaded_model.predict(sample_input)
prediction = np.argmax(output, axis=-1)[0]
# Ensure mapping is defined as before
prediction = [mapping[i] for i in prediction]
print(prediction)
