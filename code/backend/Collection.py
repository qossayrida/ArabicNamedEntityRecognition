def extract_features(tokens):
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