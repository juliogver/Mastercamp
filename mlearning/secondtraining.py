import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Charger le modèle PKL existant
with open('./nouveau_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Charger le nouveau jeu de données CSV
new_data = pd.read_csv('./Datas/test1_output.csv')

# Supprimer les lignes avec des valeurs manquantes
new_data = new_data.dropna()

# Diviser les données en caractéristiques d'entrée (X) et variable cible (y)
X_new = new_data['Review']
y_new = new_data['Sentiment']

# Charger le vectoriseur de compte à partir du fichier PKL
with open('./vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Transformer les données textuelles en caractéristiques numériques
X_new = vectorizer.transform(X_new)

# Ré-entraîner le modèle sur les nouvelles données
model.fit(X_new, y_new)

# Enregistrer le nouveau modèle entraîné
with open('nouveau1_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Calculer l'exactitude du nouveau modèle sur les nouvelles données
y_pred = model.predict(X_new)
accuracy = accuracy_score(y_new, y_pred)
print("Exactitude du nouveau modèle :", accuracy)

print("Entraînement terminé et nouveau modèle enregistré.")
