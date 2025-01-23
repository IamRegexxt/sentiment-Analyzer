import pandas as pd
import re
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from langdetect import detect
from googletrans import Translator
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# Ensure you have downloaded the stopwords
import nltk
nltk.download('stopwords')

# Initialize stop words and stemmer
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# --- 1. Load Data ---
# Load the responses, lexicon, and model parameters from CSV files
responses_df = pd.read_csv('responses1.csv')
lexicon_df = pd.read_csv('lexdict.csv')
model_df = pd.read_csv('model_parameters.csv')

# Convert lexicon to dictionary {word: sentiment_score}
lexicon = {row['word']: row['sentiment_score'] for _, row in lexicon_df.iterrows()}

# --- 2. Text Processing Functions ---

# Preprocessing: Lowercase, punctuation removal, tokenization, stop word removal, and stemming
def preprocess_text(text):
    """
    Preprocess the input text by converting to lowercase, removing punctuation,
    tokenizing, removing stop words, and performing stemming.

    Args:
        text (str): The input text to preprocess.

    Returns:
        list: A list of preprocessed tokens.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = text.split()
    tokens = [token for token in tokens if token not in stop_words]  # Remove stop words
    tokens = [stemmer.stem(token) for token in tokens]  # Perform stemming
    return tokens

# Cleansing: Remove non-alphabetic characters and rare words
def cleanse_text(tokens):
    """
    Cleanse the tokens by removing non-alphabetic characters and short words.

    Args:
        tokens (list): A list of tokens to cleanse.

    Returns:
        list: A list of cleansed tokens.
    """
    cleaned_tokens = [re.sub(r'[^a-zA-Z]', '', token) for token in tokens]
    cleaned_tokens = [token for token in cleaned_tokens if len(token) > 2]  # Remove short words
    return cleaned_tokens

# Translation: Convert Bicol or Tagalog words to English and count translated words
def translate_to_english(tokens):
    """
    Translate Bicol or Tagalog words to English and count the number of translated words.

    Args:
        tokens (list): A list of tokens to translate.

    Returns:
        tuple: A tuple containing the translated tokens, counts of Bicol and Tagalog words translated,
               total translated words, and total words needing translation.
    """
    translator = Translator()
    translated_tokens = []
    bikol_to_english_count = 0
    tagalog_to_english_count = 0
    total_translated = 0
    total_words_needing_translation = 0

    for token in tokens:
        try:
            lang = detect(token)
            if lang == 'tl':  # Tagalog
                translated_token = translator.translate(token, dest='en').text
                translated_tokens.append(translated_token.lower())
                tagalog_to_english_count += 1
                total_translated += 1
                total_words_needing_translation += 1
            elif lang == 'bk':  # Bikol
                translated_token = translator.translate(token, dest='en').text
                translated_tokens.append(translated_token.lower())
                bikol_to_english_count += 1
                total_translated += 1
                total_words_needing_translation += 1
            else:
                translated_tokens.append(token)
                total_words_needing_translation += 1
        except:
            translated_tokens.append(token)

    return translated_tokens, bikol_to_english_count, tagalog_to_english_count, total_translated, total_words_needing_translation

# Define a mapping function to convert sentiment scores to three categories
def map_sentiment_to_label(sentiment_score):
    """
    Map the sentiment score to a sentiment label ('positive', 'neutral', 'negative').

    Args:
        sentiment_score (int): The sentiment score to map.

    Returns:
        str: The sentiment label.
    """
    if sentiment_score > 0:
        return 'positive'
    elif sentiment_score == 0:
        return 'neutral'
    else:
        return 'negative'

# --- 3. Sentiment Analysis Function ---
def sentiment_analysis(tokens, lexicon):
    """
    Perform sentiment analysis on the tokens using the lexicon.

    Args:
        tokens (list): A list of tokens to analyze.
        lexicon (dict): A dictionary mapping words to sentiment scores.

    Returns:
        int: The sentiment score.
    """
    sentiment_score = sum(lexicon.get(token, 0) for token in tokens)
    return sentiment_score

# --- 4. Process Data and Perform Sentiment Analysis ---
# Combine the narratives from the responses DataFrame
narratives = responses_df[['Q9', 'Q12']].dropna()
narratives['combined'] = narratives['Q9'].astype(str) + ' ' + narratives['Q12'].astype(str)

results = []
total_bikol_translated = 0
total_tagalog_translated = 0
total_translated_words = 0
total_words_needing_translation = 0

# Process each narrative
for narrative in narratives['combined']:
    tokens = preprocess_text(narrative)
    cleaned_tokens = cleanse_text(tokens)
    translated_tokens, bikol_count, tagalog_count, total_count, total_needing_translation = translate_to_english(cleaned_tokens)
    sentiment_score = sentiment_analysis(translated_tokens, lexicon)
    sentiment_label = map_sentiment_to_label(sentiment_score)  # Map sentiment score to 'positive', 'neutral', 'negative'

    results.append({
        'original_text': narrative,
        'processed_text': ' '.join(translated_tokens),
        'sentiment_score': sentiment_score,
        'sentiment_label': sentiment_label  # Use mapped sentiment label
    })

    total_bikol_translated += bikol_count
    total_tagalog_translated += tagalog_count
    total_translated_words += total_count
    total_words_needing_translation += total_needing_translation

# Convert results to DataFrame and save
processed_df = pd.DataFrame(results)
processed_df.to_csv('narrative_sentiment.csv', index=False)

# Define a mapping dictionary to convert sentiment scores to labels
score_to_label = {
    1: 'positive',
    0: 'neutral',
    -1: 'negative'
}

# --- 5. Model Evaluation ---

# Convert model's numerical labels to string labels
true_labels = [score_to_label[1] if row['coefficient'] > 0
               else score_to_label[-1] if row['coefficient'] < 0
else score_to_label[0] for _, row in model_df.iterrows()][:len(results)]

# Ensure both true_labels and predicted_labels are only positive, neutral, or negative
predicted_labels = processed_df['sentiment_label'].values[:len(true_labels)]

# Use TF-IDF for feature representation
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(processed_df['processed_text'])

# Train an SVM classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X, true_labels)

# Predict sentiment on the processed data
y_pred = svm_classifier.predict(X)

# Confusion Matrix and Accuracy
conf_matrix = confusion_matrix(true_labels, y_pred, labels=['positive', 'neutral', 'negative'])
accuracy = accuracy_score(true_labels, y_pred)
report = classification_report(true_labels, y_pred, labels=['positive', 'neutral', 'negative'], target_names=['positive', 'neutral', 'negative'], zero_division=0)

print("Confusion Matrix:\n", conf_matrix)
print("Accuracy:", accuracy)
print("Classification Report:\n", report)

# --- 6. Visualization ---

# Sentiment Distribution Graph
def display_sentiment_distribution(predictions):
    """
    Display a bar graph of the sentiment distribution.

    Args:
        predictions (list): A list of predicted sentiment labels.
    """
    sentiment_counts = pd.Series(predictions).value_counts()
    plt.bar(sentiment_counts.index, sentiment_counts.values, color=['green', 'red', 'gray'])
    plt.xlabel('Sentiment')
    plt.ylabel('Frequency')
    plt.title('Sentiment Distribution')
    plt.show()

display_sentiment_distribution(y_pred)

# Generate Word Clouds for Sentiments
def generate_sentiment_wordcloud(processed_df):
    """
    Generate word clouds for positive and negative sentiments.

    Args:
        processed_df (DataFrame): The DataFrame containing processed text and sentiment scores.
    """
    positive_words = ' '.join(processed_df[processed_df['sentiment_score'] > 0]['processed_text'])
    negative_words = ' '.join(processed_df[processed_df['sentiment_score'] < 0]['processed_text'])

    plt.figure(figsize=(12, 6))

    # Positive sentiment word cloud
    plt.subplot(1, 2, 1)
    plt.imshow(WordCloud(width=400, height=400, background_color='white').generate(positive_words if positive_words else "No Positive Words"))
    plt.title('Positive Words')
    plt.axis('off')

    # Negative sentiment word cloud
    plt.subplot(1, 2, 2)
    plt.imshow(WordCloud(width=400, height=400, background_color='white').generate(negative_words if negative_words else "No Negative Words"))
    plt.title('Negative Words')
    plt.axis('off')

    plt.show()

generate_sentiment_wordcloud(processed_df)

# Print translation statistics
print(f"Total Bikol words translated: {total_bikol_translated}")
print(f"Total Tagalog words translated: {total_tagalog_translated}")
print(f"Total words translated: {total_translated_words}")
print(f"Translation Accuracy: {total_translated_words / total_words_needing_translation * 100:.2f}%")

# --- 7. Execution Flow (Running the pipeline automatically) ---
print("\nSentiment analysis completed successfully.")