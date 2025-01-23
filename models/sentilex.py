# -*- coding: utf-8 -*-
"""
Created on Thu May 30 04:51:58 2024
@author: Anna
"""

import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('data/multilingual-trainset.csv')


# Function to preprocess text
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    return text


# Combine Bicol, Tagalog, and English statements into a single text representation
data['combined_text'] = data['bicol'] + ' ' + data['tagalog'] + ' ' + data['english']

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data['combined_text'], data['label'], test_size=0.2,
                                                    random_state=42)

# Vectorize the text data using bag-of-words representation
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a logistic regression classifier
classifier = LogisticRegression()
classifier.fit(X_train_vec, y_train)

# Predict sentiment on the test set
y_pred = classifier.predict(X_test_vec)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

report = classification_report(y_test, y_pred, zero_division=0)
print("Classification Report:\n", report)

# Count the predicted sentiment labels
sentiment_counts = pd.Series(y_pred).value_counts()


# Display sentiment accuracy graph
def display_sentiment_accuracy_graph(accuracy_score):
    plt.bar(accuracy_score.index, accuracy_score.values)
    plt.xlabel('Sentiment')
    plt.ylabel('Frequency')
    plt.title('Sentiment Accuracy')
    plt.show()


# Display the sentiment accuracy graph
display_sentiment_accuracy_graph(sentiment_counts)


# Save the vocabulary from the CountVectorizer
# vocab_df = pd.DataFrame(vectorizer.vocabulary_.items(), columns=['word', 'index'])

# Save the model parameters (coefficients and intercept)
# model_df = pd.DataFrame({'coefficient': classifier.coef_[0]})
# model_df['index'] = model_df.index  # Add index as a column for mapping to vocabulary
# model_df['word'] = model_df['index'].map(vocab_df.set_index('index')['word'])  # Map index to vocabulary word
# model_df = model_df[['word', 'coefficient']]  # Reorder columns for clarity

# Save the model parameters to a CSV file
# model_df.to_csv('model_parameters.csv', index=False)

# Extract language and word relationships
def get_language(word, vocab, data):
    if word in data['bicol'].str.cat(sep=' ').split():
        return 'bk'
    elif word in data['tagalog'].str.cat(sep=' ').split():
        return 'tl'
    elif word in data['english'].str.cat(sep=' ').split():
        return 'en'
    return 'unknown'


# Save the vocabulary from the CountVectorizer
vocab_df = pd.DataFrame(vectorizer.vocabulary_.items(), columns=['word', 'index'])

# Create a DataFrame for model parameters
model_df = pd.DataFrame({'coefficient': classifier.coef_[0]})
model_df['index'] = model_df.index  # Add index as a column for mapping to vocabulary
model_df['word'] = model_df['index'].map(vocab_df.set_index('index')['word'])  # Map index to vocabulary word

# Map words to their language labels
model_df['language'] = model_df['word'].apply(lambda w: get_language(w, vocab_df, data))

# Reorder columns for clarity
model_df = model_df[['word', 'coefficient', 'language']]

# Save the model parameters to a CSV file
model_df.to_csv('model_parameters.csv', index=False)

# Generate word cloud for positive sentiment
positive_text = data[data['label'] == 'positive']['combined_text'].str.cat(sep=' ')
positive_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_text)

# Generate word cloud for negative sentiment
negative_text = data[data['label'] == 'negative']['combined_text'].str.cat(sep=' ')
negative_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(negative_text)

# Plot the word clouds
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(positive_wordcloud, interpolation='bilinear')
plt.title('Positive Sentiment Word Cloud')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(negative_wordcloud, interpolation='bilinear')
plt.title('Negative Sentiment Word Cloud')
plt.axis('off')

plt.show()