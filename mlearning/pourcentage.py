import csv


def calculate_rating_percentage(csv_file):
    # Les valeurs possibles de rating (flottants)
    ratings = [5.0, 4.0, 3.0, 2.0, 1.0]
    # Dictionnaire pour stocker le nombre de chaque rating
    rating_count = {r: 0 for r in ratings}

    total_count = 0  # Compteur total de ratings

    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            rating = float(row['Rate'])
            if rating in ratings:
                rating_count[rating] += 1
                total_count += 1

    rating_percentages = {r: (count / total_count) *
                          100 for r, count in rating_count.items()}

    return rating_percentages


# Utilisation de la fonction calculate_rating_percentage avec un fichier CSV appelé 'ratings.csv'
csv_file = './Datas/concatenated_new.csv'
percentage = calculate_rating_percentage(csv_file)
print(percentage)
