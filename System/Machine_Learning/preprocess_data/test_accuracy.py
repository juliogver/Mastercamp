import csv


def calculate_accuracy(predictions, labels):
    correct_predictions = 0
    total_predictions = len(predictions)

    for i in range(total_predictions):
        if predictions[i] == labels[i]:
            correct_predictions += 1

    accuracy = (correct_predictions / total_predictions) * 100
    return accuracy


def test_accuracy(csv_file):
    ratings = []
    ratings_lr = []
    ratings_nn = []

    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            ratings.append(float(row['Rating']))
            ratings_lr.append(float(row['Rating_LR']))
            ratings_nn.append(float(row['Rating_NN']))

    accuracy_lr = calculate_accuracy(ratings_lr, ratings)
    accuracy_nn = calculate_accuracy(ratings_nn, ratings)

    print(f"Accuracy (Rating_LR): {accuracy_lr}%")
    print(f"Accuracy (Rating_NN): {accuracy_nn}%")


# Exemple d'utilisation
csv_file = './System/trainings outputs/train_test_test.csv'
test_accuracy(csv_file)
