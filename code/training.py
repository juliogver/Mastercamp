import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
data = pd.read_csv('./Datas/Dataset.csv')

# Drop rows with missing values
data = data.dropna()

# Split the data into input features (X) and target variable (y)
X = data['Review']
y = data['Sentiment']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Initialize the count vectorizer
vectorizer = CountVectorizer()

# Transform the text data into numerical features
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Save the CountVectorizer object
with open('vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)

# Initialize the models
models = [
    ('Logistic Regression', LogisticRegression(max_iter=1000)),
    ('Support Vector Machine', SVC()),
    ('Random Forest', RandomForestClassifier())
]

# Train and evaluate each model
accuracies = []
model_names = []
trained_models = {}

for model_name, model in models:
    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = model.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    model_names.append(model_name)

    # Save the trained model
    with open(f'./{model_name}_model.pkl', 'wb') as file:
        pickle.dump(model, file)

print(model_names, accuracies)

# Plot the accuracies
plt.figure(figsize=(8, 6))
plt.bar(model_names, accuracies)
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Comparison of Classification Models')
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.show()
