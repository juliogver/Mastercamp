import pandas as pd

# Chemins d'accès aux fichiers CSV à concaténer
csv_files = ["./Datas/4_out.csv", "./Datas/5_out.csv",
             "./Datas/concatenated.csv"]

# Liste pour stocker les DataFrames de chaque fichier
dataframes = []

# Lecture de chaque fichier CSV et ajout du DataFrame à la liste
for file in csv_files:
    df = pd.read_csv(file)
    dataframes.append(df)

# Concaténation des DataFrames en un seul DataFrame
concatenated_df = pd.concat(dataframes)

# Chemin d'accès au fichier CSV de sortie
output_csv = "./Datas/concatenated_new.csv"

# Enregistrement du DataFrame concaténé dans un fichier CSV
concatenated_df.to_csv(output_csv, index=False)
