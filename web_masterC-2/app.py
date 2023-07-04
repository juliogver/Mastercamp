from flask import Flask, render_template, request, redirect
import pandas as pd
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import os
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
from matplotlib.colors import ListedColormap


# Set the SSL_CERT_FILE environment variable
os.environ['SSL_CERT_FILE'] = './System/Machine_Learning/cacert.pem'

nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
# Get the absolute path of the directory where the Flask application is located
app_dir = os.path.dirname(os.path.abspath(__file__))

# Specify the relative file paths within the static folder
wordcloud_all_filename = os.path.join(app_dir, 'static', 'wordcloud_all.png')
wordcloud_affin_filename = os.path.join(app_dir, 'static', 'wordcloud_affin.png')
pie_chart_filename = os.path.join(app_dir, 'static', 'pie_chart.png')
histogram_filename = os.path.join(app_dir, 'static', 'sentiment_histogram.png')

# Set colormap for limited colors
colormap = 'Set3'



# Import Afinn
from afinn import Afinn

afinn = Afinn()

def preprocess_text_normal(text):
    if isinstance(text, str):  # Check if text is a valid string
        text = text.lower()
        words = text.split()
        filtered_words = [word for word in words if word not in stop_words or word in ["not", "don't", "won't", "can't", "shouldn't", "couldn't", "wouldn't", "isn't", "aren't", "very", "just", "quite", 'about', 'above', 'across', 'after', 'against', 'along', 'among', 'around', 'as', 'at', 'before', 'behind', 'below', 'beneath', 'beside', 'between', 'beyond', 'but', 'by', 'concerning',
                                                                                       'considering', 'despite', 'down', 'during', 'except', 'for', 'from', 'in', 'inside', 'into', 'like', 'near', 'of', 'off', 'on', 'onto', 'out', 'outside', 'over', 'past', 'regarding', 'round', 'since', 'through', 'throughout', 'to', 'toward', 'under', 'underneath', 'until', 'up', 'upon', 'with', 'within', 'without']]
        sentiment_words = [word for word in filtered_words if afinn.score(word) != 0]
        text = ' '.join(sentiment_words)
        return text
    else:
        return ''  # Return empty string for non-valid valuess

def preprocess_text(text):
    if isinstance(text, str):  # Check if text is a valid string
        text = text.lower()
        words = text.split()
        filtered_words = [word for word in words if word not in stop_words]
        text = ' '.join(filtered_words)
        return text
    else:
        return '' 

# Load the Neural Network model
nn_model = load_model('./System/Machine_Learning/ia_models/neural_network_model4.h5')

# Load the TfidfVectorizer used during training
vectorizer = pickle.load(open('./System/Machine_Learning/ia_models/tfidf_vectorizer.pkl', 'rb'))


def generate_wordcloud(data):
    
  
    # Prétraitement des textes pour tous les mots
    all_text = ' '.join(data['Review'].dropna().astype(str).unique().tolist())
    preprocessed_all_text = preprocess_text(all_text)
    preprocess_affin = preprocess_text_normal(all_text)
    # Création des objets WordCloud

    wordcloud_all = WordCloud(width=800, height=400, background_color=None, colormap=colormap).generate(preprocessed_all_text)
    wordcloud_affin = WordCloud(width=800, height=400, background_color=None, colormap=colormap).generate(preprocess_affin)

    # Sauvegarde des nuages de mots en tant qu'images


  
    wordcloud_all.to_file(wordcloud_all_filename)
    wordcloud_affin.to_file(wordcloud_affin_filename)


    return  wordcloud_all_filename, wordcloud_affin_filename




def generate_sentiment_pie_chart(data):
    # Prétraitement du texte
    data['processed_text'] = data['Review'].apply(preprocess_text)

    # Transformation du texte en vecteurs avec le TfidfVectorizer
    X_vectors = vectorizer.transform(data['processed_text'])

    # Prédiction des sentiments avec le modèle de réseau de neurones
    sentiment_predictions = nn_model.predict(X_vectors.toarray())
    sentiment_predictions = [label.argmax() for label in sentiment_predictions]

    # Assigner les prédictions de sentiment au DataFrame
    data['Sentiment'] = sentiment_predictions

    # Compter les occurrences de chaque catégorie de sentiment
    sentiment_counts = data['Sentiment'].value_counts()

    # Réinitialiser le graphique précédent
    plt.clf()

    # Tracé du graphique en camembert
    labels = sentiment_counts.index
    sizes = sentiment_counts.values

    colors = ['#00ff00', '#66ff66', 'gray', '#ff6666', '#ff0000']
    explode = (0.1, 0.1, 0.1, 0.1, 0.1)

    patches, texts, autotexts = plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, pctdistance=0.85)

    # Définir la couleur du texte des étiquettes en blanc
    for text in texts:
        text.set_color('white')

    # Définir la couleur du texte des pourcentages en blanc
    for autotext in autotexts:
        autotext.set_color('white')

    # Sauvegarde du graphique en tant qu'image
    plt.savefig(pie_chart_filename, transparent=True)



def generate_sentiment_histogram(data):
    # Prétraitement du texte
    data['processed_text'] = data['Review'].apply(preprocess_text)

    # Transformation du texte en vecteurs avec le TfidfVectorizer
    X_vectors = vectorizer.transform(data['processed_text'])

    # Prédiction des sentiments avec le modèle de réseau de neurones
    sentiment_predictions = nn_model.predict(X_vectors.toarray())
    sentiment_predictions = [label.argmax() for label in sentiment_predictions]

    # Assigner les prédictions de sentiment au DataFrame
    data['Sentiment'] = sentiment_predictions

    # Compter les occurrences de chaque catégorie de sentiment
    sentiment_counts = data['Sentiment'].value_counts()

    # Réinitialiser le graphique précédent
    plt.clf()
    
    # Tracer l'histogramme des sentiments
    labels = sentiment_counts.index.sort_values(ascending=True)

    x = range(len(labels))
    heights = sentiment_counts.loc[labels].values

    bar_plot = plt.bar(x, heights, tick_label=labels, color=['#ff6666', 'gray', '#66ff66'])

    # Ajouter des étiquettes aux barres
    for i, v in enumerate(heights):
        plt.text(i, v + 10, str(v), ha='center', va='bottom', color='white')

    # Définir les titres et les étiquettes des axes en blanc
    plt.title('Histogram of Sentiments', color='white')
    plt.xlabel('Sentiment', color='white')
    plt.ylabel('Count', color='white')

    # Définir la couleur du texte des axes en blanc
    ax = plt.gca()
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    # Définir la couleur des traits d'axe en blanc
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')

    # Définir la couleur de fond du graphique sur une couleur autre que blanc
    ax.set_facecolor('#222222')

    # Sauvegarder le graphique en tant qu'image
    plt.savefig(histogram_filename, transparent=True)

  




app = Flask(__name__)


@app.route('/')
def home():
    global data 
    data = None
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect('/')
    data = None
    file = request.files['file']

    # Lire le fichier CSV en utilisant pandas
    data = pd.read_csv(file)

   
    # Générer le nuage de mots
    generate_wordcloud(data)

    # Générer le graphique circulaire des sentiments
    generate_sentiment_pie_chart(data)

    generate_sentiment_histogram(data)

    # Rediriger vers la page de résultat
    return redirect('/result')

    


@app.route('/result')
def result():
    return render_template('result.html')


if __name__ == '__main__':
    app.run(debug=True)
