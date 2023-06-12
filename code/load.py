import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import pickle

# Load the dataset
data = pd.read_csv('./Datas/Dataset.csv')

# Drop rows with missing values
data = data.dropna()

# Split the data into input features (X) and target variable (y)
X = data['Review']
y = data['Sentiment']

# Load the vectorizer
with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Transform the text data into numerical features
X = vectorizer.transform(X)

# Load the models
model_names = ['Logistic Regression',
               'Support Vector Machine', 'Random Forest']
models = []
for model_name in model_names:
    with open(f'{model_name}_model.pkl', 'rb') as file:
        model = pickle.load(file)
        models.append(model)

# Predict sentiment on new data
new_reviews = ["This product is great!",
               "I'm very disappointed with this purchase."]
X_new = vectorizer.transform(new_reviews)

for model_name, model in zip(model_names, models):
    # Make predictions on the new data
    y_pred = model.predict(X_new)

    # Print the predicted sentiment for each review
    print(f"Predictions using {model_name}:")
    for review, sentiment in zip(new_reviews, y_pred):
        print(f"Review: {review}")
        print(f"Sentiment: {sentiment}")
        print()
