import pandas as pd

# Chemin d'accès au fichier CSV
csv_file = "./Datas/Dataset.csv"

# Listes de mots synonymes de "Rating", "Commentary" et "Summary"
rating_synonyms = ["Rating", "Grade", "Evaluation",
                   "Score", "Rate", "Mark", "Rank", "Ranking", "Stars"]
commentary_synonyms = ["Commentary", "Review", "Feedback",
                       "Opinion", "Comment", "Criticism", "Appreciation", "Appraisal", "Translated"]
summary_synonyms = ["Summary", "Synopsis",
                    "Abstract", "Overview", "Description"]

# Convertir tous les synonymes en minuscules pour une correspondance insensible à la casse
rating_synonyms = [syn.lower() for syn in rating_synonyms]
commentary_synonyms = [syn.lower() for syn in commentary_synonyms]
summary_synonyms = [syn.lower() for syn in summary_synonyms]

# Lecture du fichier CSV
df = pd.read_csv(csv_file)

# Convertir les noms de colonnes en minuscules pour une correspondance insensible à la casse
df.columns = [col.lower() for col in df.columns]

# Recherche des colonnes correspondant aux synonymes de "Rating", "Commentary" et "Summary"
rating_columns = [col for col in df.columns if any(
    syn in col.lower().split() for syn in rating_synonyms)]
commentary_columns = [col for col in df.columns if any(
    syn in col.lower().split() for syn in commentary_synonyms)]
summary_columns = [col for col in df.columns if any(
    syn in col.lower().split() for syn in summary_synonyms)]

# Sélection des colonnes d'intérêt
columns_of_interest = rating_columns + commentary_columns + summary_columns

# Création du nouveau DataFrame avec les colonnes sélectionnées
new_df = df[columns_of_interest].copy()

# Renommer les colonnes en "Rate", "Review", "Summary" et "Sentiment"
new_df = new_df.rename(columns=dict(
    zip(columns_of_interest, ["Rate", "Review", "Summary", "Sentiment"])))

# Suppression des lignes où la colonne "Review" est vide
new_df = new_df.dropna(subset=["Review"])

# Convertir la colonne "Rate" en entier


def convert_rate(rate):
    try:
        return int(rate)
    except ValueError:
        return None


new_df["Rate"] = new_df["Rate"].apply(convert_rate)

# Suppression des lignes avec des valeurs non numériques dans la colonne "Rate"
new_df = new_df.dropna(subset=["Rate"])

# Création de la colonne "Sentiment" en fonction de la colonne "Rate"


def map_sentiment(rate):
    if rate == 5:
        return "very positive"
    elif rate == 4:
        return "positive"
    elif rate == 3:
        return 'neutral'
    elif rate == 2:
        return "negative"
    else:
        return "very negative"


new_df["Sentiment"] = new_df["Rate"].apply(map_sentiment)

# Enregistrement du nouveau DataFrame dans un fichier CSV
new_csv_file = "output.csv"
new_df.to_csv(new_csv_file, index=False)
