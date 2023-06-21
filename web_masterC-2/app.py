from flask import Flask, render_template, request, redirect
import pandas as pd
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')  # Ajouter cette ligne avant l'import de pyplot
import matplotlib.pyplot as plt
import os
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

# Set the SSL_CERT_FILE environment variable
os.environ['SSL_CERT_FILE'] = './cacert.pem'

nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
# Get the absolute path of the directory where the Flask application is located
app_dir = os.path.dirname(os.path.abspath(__file__))
# Specify the relative file paths within the static folder
wordcloud_filename = os.path.join(app_dir, 'static', 'wordcloud.png')
pie_chart_filename = os.path.join(app_dir, 'static', 'pie_chart.png')
def preprocess_text(text):
    if isinstance(text, str):  # Check if text is a valid string
        text = text.lower()
        words = text.split()
        filtered_words = [word for word in words if word not in stop_words or word in ["not", "don't", "won't", "can't", "shouldn't", "couldn't", "wouldn't", "isn't", "aren't", "very", "just", "quite", 'about', 'above', 'across', 'after', 'against', 'along', 'among', 'around', 'as', 'at', 'before', 'behind', 'below', 'beneath', 'beside', 'between', 'beyond', 'but', 'by', 'concerning',
                                                                                       'considering', 'despite', 'down', 'during', 'except', 'for', 'from', 'in', 'inside', 'into', 'like', 'near', 'of', 'off', 'on', 'onto', 'out', 'outside', 'over', 'past', 'regarding', 'round', 'since', 'through', 'throughout', 'to', 'toward', 'under', 'underneath', 'until', 'up', 'upon', 'with', 'within', 'without']]
        text = ' '.join(filtered_words)
        return text
    else:
        return ''  # Return empty string for non-valid values


# Load the Neural Network model
nn_model = load_model('./System/ia/ia_models/neural_network_model4.h5')

# Load the TfidfVectorizer used during training
vectorizer = pickle.load(open('./System/ia/ia_models/tfidf_vectorizer.pkl', 'rb'))


def generate_wordcloud(data):
    # Concaténer les valeurs uniques de la colonne 'Review' en une seule chaîne de caractères
    text = ' '.join(data['Comment'].dropna().astype(str).unique().tolist())

    # Prétraitement du texte
    preprocessed_text = preprocess_text(text)

    # Créer l'objet WordCloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(preprocessed_text)

    # Sauvegarder le nuage de mots en tant qu'image
  
    wordcloud.to_file(wordcloud_filename)


def generate_sentiment_pie_chart(data):
    # Prétraitement du texte
    data['processed_text'] = data['Comment'].apply(preprocess_text)

    # Transformation du texte en vecteurs avec le TfidfVectorizer
    X_vectors = vectorizer.transform(data['processed_text'])

    # Prédiction des sentiments avec le modèle de réseau de neurones
    sentiment_predictions = nn_model.predict(X_vectors.toarray())
    sentiment_predictions = [label.argmax() for label in sentiment_predictions]

    # Assigner les prédictions de sentiment au DataFrame
    data['Sentiment'] = sentiment_predictions

    # Compter les occurrences de chaque catégorie de sentiment
    sentiment_counts = data['Sentiment'].value_counts()

    # Plot du graphique en camembert
    labels = sentiment_counts.index
    sizes = sentiment_counts.values

    colors = ['#00ff00', '#66ff66', 'gray', '#ff6666', '#ff0000']
    explode = (0.1, 0, 0, 0, 0)

    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')

    
    plt.savefig(pie_chart_filename)


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

    # Générer le nuage de mots
    generate_wordcloud(data)

    # Générer le graphique circulaire des sentiments
    generate_sentiment_pie_chart(data)

    # Rediriger vers la page de résultat
    return redirect('/result')


@app.route('/result')
def result():
    return render_template('result.html')


if __name__ == '__main__':
    app.run(debug=True)
