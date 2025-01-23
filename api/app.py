# -*- coding: utf-8 -*-
"""
Created on Thu May 30 04:51:58 2024
@author: Anna
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import re
from googletrans import Translator
from langdetect import detect
import numpy as np

app = Flask(__name__)
CORS(app)

# Load lexicon and model parameters
lexicon_df = pd.read_csv('data/lexdict.csv')
model_df = pd.read_csv('data/model_parameters.csv')

# Create dictionaries
lexicon = {row['word']: row['sentiment_score'] for _, row in lexicon_df.iterrows()}
model_params = {row['word']: row['coefficient'] for _, row in model_df.iterrows()}

# Initialize translator
translator = Translator()


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text


def translate_tokens(tokens):
    translated = []
    bikol_count = 0
    tagalog_count = 0
    total_translated = 0

    for token in tokens:
        try:
            lang = detect(token)
            if lang == 'tl':
                translated.append(translator.translate(token, dest='en').text.lower())
                tagalog_count += 1
                total_translated += 1
            elif lang == 'bcl':  # Bikol language code
                translated.append(translator.translate(token, dest='en').text.lower())
                bikol_count += 1
                total_translated += 1
            else:
                translated.append(token)
        except:
            translated.append(token)

    return translated, bikol_count, tagalog_count, total_translated


def analyze_sentiment(text):
    # Preprocess
    tokens = preprocess_text(text).split()

    # Translate
    translated_tokens, bikol_cnt, tl_cnt, total_trans = translate_tokens(tokens)

    # Calculate lexicon score
    lexicon_score = sum(lexicon.get(token, 0) for token in translated_tokens)

    # Calculate model score
    model_score = sum(model_params.get(token, 0) for token in translated_tokens)

    # Combine scores
    combined_score = lexicon_score + model_score

    # Determine sentiment
    if combined_score > 0:
        label = 'positive'
    elif combined_score < 0:
        label = 'negative'
    else:
        label = 'neutral'

    return {
        'sentiment_score': float(combined_score),
        'sentiment_label': label,
        'translation_stats': {
            'bikol_translated': bikol_cnt,
            'tagalog_translated': tl_cnt,
            'total_translated': total_trans,
            'total_words': len(tokens)
        }
    }


@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    try:
        result = analyze_sentiment(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)