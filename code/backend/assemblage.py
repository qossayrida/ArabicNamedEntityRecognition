import re
def load_data(file):
    data = []  # To store (sentence, labels) tuples
    sentence = []
    labels = []

    with open(file, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()

            # Split sentences at empty lines
            if not line:
                if sentence:  # If we have a completed sentence
                    data.append((" ".join(sentence), labels))
                    sentence = []
                    labels = []
                continue

            # Split the word and its label
            word, label = line.rsplit(' ', 1)
            sentence.append(word)
            labels.append(label)

    # Add the last sentence if file doesn't end with a blank line
    if sentence:
        data.append((" ".join(sentence), labels))

    return data

def extract_features_for_crf(tokens):
    features = []
    for i, token in enumerate(tokens):
        token_features = {
            'word': token,
            'is_digit': token.isdigit(),
            'prefix1': token[:1],
            'suffix1': token[-1:],
            'is_arabic': all('\u0600' <= char <= '\u06FF' for char in token),
        }
        if i > 0:
            token_features['prev_word'] = tokens[i - 1]
        else:
            token_features['prev_word'] = '<START>'
        if i < len(tokens) - 1:
            token_features['next_word'] = tokens[i + 1]
        else:
            token_features['next_word'] = '<END>'
        features.append(token_features)
    return features


def extract_features_for_DT_NB(sentence, idx):
    word = sentence[idx]
    features = {
        "word": word,
        "is_first": idx == 0,
        "is_last": idx == len(sentence) - 1,
        "prefix-1": word[:1],
        "prefix-2": word[:2],
        "prefix-3": word[:3],
        "suffix-1": word[-1:],
        "suffix-2": word[-2:],
        "suffix-3": word[-3:],
        "is_numeric": word.isdigit(),
        "contains_digit": any(char.isdigit() for char in word),
        "word_shape": re.sub(r"[\u0621-\u064A]", "X", re.sub(r"\d", "d", word)),
    }
    return features




