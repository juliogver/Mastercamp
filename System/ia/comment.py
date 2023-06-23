import pandas as pd

# Lire le fichier CSV
data = pd.read_csv("./System/Datas/amazon_review.csv")

# Garder uniquement la colonne "comment"
data = data[["Review"]]

# Enregistrer les données dans un nouveau fichier CSV
data.to_csv("./System/Datas/only_review.csv", index=False)
