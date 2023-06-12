import pandas as pd
from nltk.corpus import wordnet

# Chemin d'accès au fichier CSV
csv_file = "./Datas/test.csv"

# Listes de mots synonymes de "Rating" et "Commentary"
rating_synonyms = ["Rating", "Grade", "Evaluation",
                   "Score", "Rate", "Mark", "Rank", "Ranking", "Stars"]
commentary_synonyms = ["Commentary", "Review", "Feedback",
                       "Opinion", "Comment", "Criticism", "Appreciation", "Appraisal", "Translated"]

# Convertir tous les synonymes en minuscules pour une correspondance insensible à la casse
rating_synonyms = [syn.lower() for syn in rating_synonyms]
commentary_synonyms = [syn.lower() for syn in commentary_synonyms]

# Lecture du fichier CSV
df = pd.read_csv(csv_file)

# Convertir les noms de colonnes en minuscules pour une correspondance insensible à la casse
df.columns = [col.lower() for col in df.columns]

# Recherche des colonnes correspondant aux synonymes de "Rating" et "Commentary"
rating_columns = [col for col in df.columns if any(
    syn in col.lower().split() for syn in rating_synonyms)]
commentary_columns = [col for col in df.columns if any(
    syn in col.lower().split() for syn in commentary_synonyms)]

# Sélection des colonnes d'intérêt
columns_of_interest = rating_columns + commentary_columns

# Création du nouveau DataFrame avec les colonnes sélectionnées
new_df = df[columns_of_interest].copy()

# Renommer les colonnes en "Rate", "Review", et "Sentiment"
new_df = new_df.rename(columns=dict(
    zip(columns_of_interest, ["Rate", "Review", "Sentiment"])))

# Suppression des lignes où la colonne "Review" est vide
new_df = new_df.dropna(subset=["Review"])

# Création de la colonne "Sentiment" en fonction de la colonne "Rate"
new_df["Sentiment"] = new_df["Rate"].apply(
    lambda x: "very positive" if x == 5 else "positive" if x == 4 else "neutral" if x == 3 else "negative" if x == 2 else "very negative")

# Enregistrement du nouveau DataFrame dans un fichier CSV
new_csv_file = "output.csv"
new_df.to_csv(new_csv_file, index=False)
