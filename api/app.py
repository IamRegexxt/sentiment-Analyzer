# -*- coding: utf-8 -*-
"""
jez
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import re
import langid
from googletrans import Translator
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

app = Flask(__name__)
CORS(app)

# Initialize NLP components
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
translator = Translator()

# Load data files
lexicon_df = pd.read_csv('data/lexdict.csv')
lexicon = {row['word']: row['sentiment_score'] for _, row in lexicon_df.iterrows()}


def detect_language(text):
    try:
        return langid.classify(text)[0]
    except:
        return 'en'


def translate_to_english(text):
    try:
        lang = detect_language(text)
        if lang in ['tl', 'bik']:
            return translator.translate(text, dest='en').text.lower()
        return text.lower()
    except Exception as e:
        print(f"Translation error: {str(e)}")
        return text.lower()


def process_text(text):
    # Clean and translate
    cleaned = re.sub(r'[^\w\s]', '', text).lower()
    translated = translate_to_english(cleaned)

    # Tokenize and process
    tokens = translated.split()
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [stemmer.stem(t) for t in tokens]

    return tokens


def analyze_sentiment(tokens):
    score = 0
    translation_stats = {
        'bikol_translated': 0,
        'tagalog_translated': 0,
        'total_translated': 0,
        'total_words': len(tokens)
    }

    for token in tokens:
        if token in lexicon:
            score += lexicon[token]
        else:
            try:
                # Attempt to translate unknown words
                lang = detect_language(token)
                if lang == 'bk':
                    translation_stats['bikol_translated'] += 1
                    translation_stats['total_translated'] += 1
                elif lang == 'tl':
                    translation_stats['tagalog_translated'] += 1
                    translation_stats['total_translated'] += 1

                translated = translator.translate(token, dest='en').text.lower()
                score += lexicon.get(translated, 0)
            except:
                continue

    return score, translation_stats


@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    try:
        tokens = process_text(text)
        score, stats = analyze_sentiment(tokens)

        label = 'positive' if score > 0 else 'negative' if score < 0 else 'neutral'

        return jsonify({
            'sentiment_label': label,
            'sentiment_score': score,
            'translation_stats': stats
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)