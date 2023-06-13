import nltk
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

# Load the dataset
df = pd.read_csv('./Datas/output.csv')

# Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    text = text.lower()
    text = [lemmatizer.lemmatize(word)
            for word in text.split() if word not in stop_words]
    return ' '.join(text)


df['processed_text'] = df['Review'].apply(preprocess_text)

# Convert sentiment into binary labels
df['Sentiment'] = df['Sentiment'].map(
    {'very positive': 5, 'positive': 4, 'neutral': 3, 'negative': 2, 'very negative': 1})

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df['processed_text'], df['Sentiment'], test_size=0.2, random_state=42)

# Vectorize the text with n-grams
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Save the vectorizer
pickle.dump(vectorizer, open('tfidf_vectorizer.pkl', 'wb'))

# Train the Logistic Regression model
clf_lr = LogisticRegression()
clf_lr.fit(X_train_vectors, y_train)

# Save the Logistic Regression model
pickle.dump(clf_lr, open('logistic_regression_model.pkl', 'wb'))

# Evaluate the Logistic Regression model
y_pred_lr = clf_lr.predict(X_test_vectors)
print(classification_report(y_test, y_pred_lr))

# Prepare data for neural network
encoder = LabelEncoder()
encoder.fit(y_train)
encoded_Y_train = encoder.transform(y_train)
dummy_y_train = np_utils.to_categorical(encoded_Y_train)

encoded_Y_test = encoder.transform(y_test)
dummy_y_test = np_utils.to_categorical(encoded_Y_test)

# Create a neural network
clf_nn = Sequential()
clf_nn.add(Dense(32, input_dim=X_train_vectors.shape[1], activation='relu'))
clf_nn.add(Dense(5, activation='softmax'))
clf_nn.compile(loss='categorical_crossentropy',
               optimizer='adam', metrics=['accuracy'])

# Train the neural network
clf_nn.fit(X_train_vectors.toarray(), dummy_y_train, epochs=10, batch_size=64)

# Save the neural network
clf_nn.save('neural_network_model.h5')

# Evaluate the neural network
loss, accuracy = clf_nn.evaluate(X_test_vectors.toarray(), dummy_y_test)
print('Accuracy: %.2f' % (accuracy*100))
