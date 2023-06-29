import os
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.utils import to_categorical
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

# Set the SSL_CERT_FILE environment variable
os.environ['SSL_CERT_FILE'] = './System/modeling/cacert.pem'

# Load the models
vectorizer = pickle.load(open('./System/Machine_Learning/ia_models/tfidf_vectorizer.pkl', 'rb'))
clf_lr = pickle.load(
    open('./System/Machine_Learning/ia_models/logistic_regression_model3.pkl', 'rb'))
clf_nn = load_model('./System/Machine_Learning/ia_models/neural_network_model3.h5')

# Load the new dataset
new_data = pd.read_csv('./Datas/test_out_out.csv')

# Ajoute les mots à exclure dans la liste des stopwords
stop_words = set(stopwords.words('english'))
excluded_words = ["not", "don't", "won't", "can't", "shouldn't", "couldn't", "wouldn't", "isn't", "aren't", "very", "just", "quite", 'about', 'above', 'across', 'after', 'against', 'along', 'among', 'around', 'as', 'at', 'before', 'behind', 'below', 'beneath', 'beside', 'between', 'beyond', 'but', 'by', 'concerning',
                  'considering', 'despite', 'down', 'during', 'except', 'for', 'from', 'in', 'inside', 'into', 'like', 'near', 'of', 'off', 'on', 'onto', 'out', 'outside', 'over', 'past', 'regarding', 'round', 'since', 'through', 'throughout', 'to', 'toward', 'under', 'underneath', 'until', 'up', 'upon', 'with', 'within', 'without']
stop_words = stop_words.difference(excluded_words)

# ...


def preprocess_text(text):
    if isinstance(text, str):  # Check if text is a valid string
        text = text.lower()
        words = text.split()
        filtered_words = [word for word in words if word not in stop_words]
        text = ' '.join(filtered_words)
        return text
    else:
        return ''  # Return empty string for non-valid values


new_data['processed_text'] = new_data['Review'].apply(preprocess_text)
new_data['combined_text'] = new_data['processed_text']

# Vectorize the text in the new dataset
X_new_vectors = vectorizer.transform(new_data['combined_text'])

# Map sentiment labels to numeric values
sentiment_mapping = {
    'very positive': 5,
    'positive': 4,
    'neutral': 3,
    'negative': 2,
    'very negative': 1
}
new_data['Sentiment'] = new_data['Sentiment'].map(sentiment_mapping)

# Update the Logistic Regression model
clf_lr.fit(X_new_vectors, new_data['Sentiment'])

# Update the Neural Network model
num_classes = len(sentiment_mapping) + 1  # Add 1 for the new class
clf_nn = Sequential()
clf_nn.add(Dense(32, input_dim=X_new_vectors.shape[1], activation='relu'))
clf_nn.add(Dense(64, activation='relu'))
clf_nn.add(Dense(128, activation='relu'))
clf_nn.add(Dense(num_classes, activation='softmax'))
clf_nn.compile(loss='categorical_crossentropy',
               optimizer='adam', metrics=['accuracy'])
clf_nn.fit(X_new_vectors.toarray(), to_categorical(
    new_data['Sentiment'], num_classes=num_classes), epochs=150, batch_size=64)

# Save the updated models
pickle.dump(clf_lr, open('./System/Machine_Learning/ia_models/logistic_regression_model4.pkl', 'wb'))
clf_nn.save('./System/Machine_Learning/ia_models/neural_network_model4.h5')

# Load the test dataset
test_data = new_data

test_data = test_data.dropna(subset=['Sentiment'])


# Preprocess the test data
test_data['processed_text'] = test_data['Review'].apply(preprocess_text)
test_data['combined_text'] = test_data['processed_text']

# Vectorize the test data using the same vectorizer
X_test_vectors = vectorizer.transform(test_data['combined_text'])

# Map sentiment labels to numeric values
test_data['Sentiment'] = test_data['Sentiment'].map(sentiment_mapping)

# Evaluate the Logistic Regression model
accuracy_lr = clf_lr.score(X_test_vectors, test_data['Sentiment'])
print("Logistic Regression Model Accuracy:", accuracy_lr)

# Evaluate the Neural Network model
loss, accuracy_nn = clf_nn.evaluate(X_test_vectors.toarray(), to_categorical(
    test_data['Sentiment'], num_classes=num_classes))
print("Neural Network Model Accuracy:", accuracy_nn)