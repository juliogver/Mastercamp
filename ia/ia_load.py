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
clf_lr = pickle.load(
    open('./ia/ia_models/logistic_regression_model3.pkl', 'rb'))
clf_nn = load_model('./ia/ia_models/neural_network_model3.h5')

# Load the vectorizer
vectorizer = pickle.load(open('./ia/ia_models/tfidf_vectorizer.pkl', 'rb'))

# Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    text = text.lower()
    text = [lemmatizer.lemmatize(word)
            for word in text.split() if word not in stop_words]
    return ' '.join(text)


# Input comments
comments = ['This is a great product!',
            'I hate this product. It is horrible!',
            'This product is okay.',
            'Very bad!',
            'Good but not exceptional.',
            'I absolutely love this product!',
            'I am extremely disappointed with this product.',
            'This product exceeded my expectations.',
            'I have mixed feelings about this product.',
            'This is the worst product I have ever purchased.']

# Preprocess the comments
comments = [preprocess_text(comment) for comment in comments]

# Vectorize the comments
comment_vectors = vectorizer.transform(comments)

# Make predictions with the Logistic Regression model
predictions_lr = clf_lr.predict(comment_vectors)

# Make predictions with the Neural Network
predictions_nn = clf_nn.predict(comment_vectors.toarray())
predictions_nn_classes = np.argmax(predictions_nn, axis=1)


# Labels
# Exemple d'étiquettes réelles correspondant aux commentaires
labels = [5, 1, 3, 0, 4, 5, 1, 5, 2, 0]

# Comparaison avec les étiquettes réelles
for comment, prediction_lr, prediction_nn, label in zip(comments, predictions_lr, predictions_nn_classes, labels):
    print(f'Comment: {comment}')
    print(f'Logistic Regression Prediction: {prediction_lr}')
    print(f'Neural Network Prediction: {prediction_nn}')
    print(f'Actual Label: {label}')
    print()
