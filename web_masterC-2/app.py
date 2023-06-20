from flask import Flask, render_template, request, redirect
import pandas as pd
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')  # Ajouter cette ligne avant l'import de pyplot

import matplotlib.pyplot as plt


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect('/')
    
    file = request.files['file']
    
    # Lire le fichier CSV en utilisant pandas
    data = pd.read_csv(file)
    
    # Traiter les données et générer les visualisations
    generate_wordcloud(data)
    generate_sentiment_pie_chart(data)
    
    # Rediriger vers la page de résultat
    return redirect('/result')


def generate_wordcloud(data):
    # Concaténer les valeurs de la colonne 'Review' en une seule chaîne de caractères
    text = ' '.join(data['Review'].dropna().astype(str).tolist())

    # Créer l'objet WordCloud
    wordcloud = WordCloud(width=400, height=200, background_color='white').generate(text)

    # Sauvegarder le nuage de mots en tant qu'image
    wordcloud_filename = './templates/wordcloud.png'
    wordcloud.to_file(wordcloud_filename)


def generate_sentiment_pie_chart(data):
    # Supprimer les lignes avec des valeurs manquantes dans la colonne "Sentiment"
    data = data.dropna(subset=['Sentiment'])

    # Compter les occurrences de chaque catégorie de sentiment
    sentiment_counts = data['Sentiment'].value_counts()

    # Plot du graphique en camembert
    labels = sentiment_counts.index
    sizes = sentiment_counts.values

    colors = ['#00ff00', '#66ff66', 'gray', '#ff6666', '#ff0000']
    explode = (0.1, 0, 0, 0, 0)

    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')

    # Sauvegarder le graphique en tant qu'image
    pie_chart_filename = './templates/pie_chart.png'
    plt.savefig(pie_chart_filename)


@app.route('/result')
def result():
    # Récupérer les noms des fichiers des images générées
    wordcloud_filename = 'wordcloud.png'
    pie_chart_filename = 'pie_chart.png'

    return render_template('result.html', wordcloud_filename=wordcloud_filename, pie_chart_filename=pie_chart_filename)


if __name__ == '__main__':
    app.run(debug=True)
import os
from flask import Flask, render_template, request, redirect

import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

app = Flask(__name__)

def create_word_cloud(csv_file):
    # Read the CSV file using pandas
    data = pd.read_csv(csv_file)

    # Concatenate the values of the 'Review' column into a single string
    text = ' '.join(data['Review'].dropna().astype(str).tolist())

    # Create the WordCloud object
    wordcloud = WordCloud(width=400, height=200, background_color='white').generate(text)

    # Create a subplot for the word cloud
    plt.subplot(1, 2, 1)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud')

def generate_sentiment_pie_chart(data):
    # Remove rows with missing values in the "Sentiment" column
    data = data.dropna(subset=['Sentiment'])

    # Count the occurrences of each sentiment category
    sentiment_counts = data['Sentiment'].value_counts()

    # Plot du graphique en camembert
    labels = sentiment_counts.index
    sizes = sentiment_counts.values
    colors = ['#00ff00', '#66ff66', 'gray', '#ff6666', '#ff0000']
    explode = (0.1, 0, 0, 0, 0)

    plt.switch_backend('Agg')  # Changer le backend de Matplotlib

    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')

    # Sauvegarder le graphique en tant qu'image
    pie_chart_filename = 'pie_chart.png'
    plt.savefig(pie_chart_filename)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect('/')
    
    file = request.files['file']
    if file.filename == '':
        return redirect('/')

    # Sauvegarder le fichier CSV
    csv_file = 'uploaded.csv'
    file.save(csv_file)

    # Chargement des données CSV
    data = pd.read_csv(csv_file)

    # Génération du Word Cloud
    create_word_cloud(csv_file)

    # Génération du pie chart des sentiments
    generate_sentiment_pie_chart(data)

    # Rediriger vers la page de résultats
    return redirect('/result')

@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)
