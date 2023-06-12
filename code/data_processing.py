import pandas as pd
import string

# Chargement du fichier CSV
df = pd.read_csv('./Datas/Dataset.csv')

# Affichage du nombre de lignes du fichier CSV original
print("Nombre de lignes du fichier CSV original:", len(df))

# Suppression des doublons basés sur la colonne "Review"
df = df.drop_duplicates(subset=['Review'])

# Suppression de la colonne "Summary"
df = df.drop('Summary', axis=1)

# Suppression des ponctuations (à l'exception des virgules pour les colonnes)
df = df.applymap(lambda x: x.translate(str.maketrans(
    "", "", string.punctuation.replace(",", ""))) if isinstance(x, str) else x)

# Attribution des sentiments en fonction de la colonne "Rate"
sentiment_mapping = {
    '1': 'very negative',
    '2': 'negative',
    '3': 'neutral',
    '4': 'positive',
    '5': 'very positive'
}
df['Sentiment'] = df['Rate'].astype(str).map(
    lambda x: sentiment_mapping.get(x, 'unknown'))

# Suppression des lignes où la colonne "Rate" est vide
df = df.dropna(subset=['Rate'])

# Affichage du nombre de lignes du DataFrame modifié
print("Nombre de lignes du DataFrame modifié:", len(df))

# Enregistrement du DataFrame modifié dans un nouveau fichier CSV
df.to_csv('./Datas/Dataset_modified.csv', index=False)
