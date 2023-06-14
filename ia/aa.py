import pandas as pd
import pickle
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

# Set the SSL_CERT_FILE environment variable
import os
os.environ['SSL_CERT_FILE'] = './cacert.pem'

nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    if isinstance(text, str):  # Check if text is a valid string
        text = text.lower()
        words = text.split()
        filtered_words = [word for word in words if word not in stop_words or word in ["not", "don't", "won't", "can't", "shouldn't", "couldn't", "wouldn't", "isn't", "aren't", "very", "just", "quite", 'about', 'above', 'across', 'after', 'against', 'along', 'among', 'around', 'as', 'at', 'before', 'behind', 'below', 'beneath', 'beside', 'between', 'beyond', 'but', 'by', 'concerning',
                                                                                       'considering', 'despite', 'down', 'during', 'except', 'for', 'from', 'in', 'inside', 'into', 'like', 'near', 'of', 'off', 'on', 'onto', 'out', 'outside', 'over', 'past', 'regarding', 'round', 'since', 'through', 'throughout', 'to', 'toward', 'under', 'underneath', 'until', 'up', 'upon', 'with', 'within', 'without']]
        text = ' '.join(filtered_words)
        return text
    else:
        return ''  # Return empty string for non-valid values

 # Return empty string for non-valid values
# Load the Logistic Regression model
clf_lr = pickle.load(
    open('./ia/ia_models/logistic_regression_model4.pkl', 'rb'))

# Load the Neural Network model
clf_nn = load_model('./ia/ia_models/neural_network_model4.h5')

# Load the TfidfVectorizer used during training
vectorizer = pickle.load(open('./ia/ia_models/tfidf_vectorizer.pkl', 'rb'))

# Load the CSV file
df = pd.read_csv('./Datas/test_out_out.csv')

# Preprocess the comments
df['processed_text'] = df['Comment'].apply(preprocess_text)

# Transform the preprocessed comments using the TfidfVectorizer
X_vectors = vectorizer.transform(df['processed_text'])

# Predict sentiment using Logistic Regression
lr_predictions = clf_lr.predict(X_vectors)

# Predict sentiment using Neural Network
nn_predictions = clf_nn.predict(X_vectors.toarray())
nn_predictions = [label.argmax() for label in nn_predictions]

# Assign ratings based on sentiment predictions
df['Rating_LR'] = lr_predictions
df['Rating_NN'] = nn_predictions

# Save the ratings to a new CSV file
df.to_csv('./trainings outputs/train_test_test.csv', index=False)
