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
    # Concaténer les valeurs uniques de la colonne 'Review' en une seule chaîne de caractères
    text = ' '.join(data['Review'].dropna().astype(str).unique().tolist())

    # Créer l'objet WordCloud
    wordcloud = WordCloud(width=400, height=200, background_color='white').generate(text)

    # Sauvegarder le nuage de mots en tant qu'image
    wordcloud_filename = 'C:\\Users\\enzoc\\OneDrive\\Documents\\Mastercamp\\web_masterC-2\\static\\wordcloud.png'
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
    pie_chart_filename = 'C:\\Users\\enzoc\\OneDrive\\Documents\\Mastercamp\\web_masterC-2\\static\\pie_chart.png'
    plt.savefig(pie_chart_filename)


@app.route('/result')
def result():
    # Récupérer les noms des fichiers des images générées
    wordcloud_filename = 'static/wordcloud.png'
    pie_chart_filename = 'static/pie_chart.png'


    return render_template('result.html', wordcloud_filename=wordcloud_filename, pie_chart_filename=pie_chart_filename)


if __name__ == '__main__':
    app.run(debug=True)
