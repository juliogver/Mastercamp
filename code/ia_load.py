import pandas as pd
import numpy as np
import pickle
from keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

# Load the models
clf_lr = pickle.load(open('./logistic_regression_model.pkl', 'rb'))
clf_nn = load_model('./neural_network_model.h5')

# Load the vectorizer
vectorizer = pickle.load(open('./tfidf_vectorizer.pkl', 'rb'))

# Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    text = text.lower()
    text = [lemmatizer.lemmatize(word)
            for word in text.split() if word not in stop_words]
    return ' '.join(text)


# Input comments
comments = ['this is a great product',
            'I hate this product it is horrible', 'this product is okay', 'very bad', 'good but not exeptional']

# Preprocess the comments
comments = [preprocess_text(comment) for comment in comments]

# Vectorize the comments
comment_vectors = vectorizer.transform(comments)

# Make predictions with the Logistic Regression model
predictions_lr = clf_lr.predict(comment_vectors)

# Make predictions with the Neural Network
predictions_nn = clf_nn.predict(comment_vectors.toarray())
predictions_nn_classes = np.argmax(predictions_nn, axis=1)
# Print predictions
for comment, prediction_lr, prediction_nn in zip(comments, predictions_lr, predictions_nn_classes):
    print(f'Comment: {comment}')
    print(f'Logistic Regression Prediction: {prediction_lr}')
    print(f'Neural Network Prediction: {prediction_nn}')
    print()
