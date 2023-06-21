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
wordcloud_all_filename = os.path.join(app_dir, 'static', 'wordcloud_all.png')
wordcloud_negative_filename = os.path.join(app_dir, 'static', 'wordcloud_negative.png')
wordcloud_positive_filename= os.path.join(app_dir, 'static', 'wordcloud_postive.png')
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
    
    sentiment_mapping = {
        'very negative': 1,
        'negative': 2,
        'neutral': 3,
        'positive': 4,
        'very positive': 5
    }

    data['Sentiment'] = data['Sentiment'].map(sentiment_mapping)
    # Prétraitement des textes pour les sentiments positifs
    positive_text = ' '.join(data[data['Sentiment'].astype(int) >= 4]['Review'].dropna().astype(str).unique().tolist())
    
    preprocessed_positive_text = preprocess_text(positive_text)

    # Prétraitement des textes pour les sentiments négatifs
    negative_text = ' '.join(data[data['Sentiment'].astype(int) <= 2]['Review'].dropna().astype(str).unique().tolist())

    preprocessed_negative_text = preprocess_text(negative_text)

    # Prétraitement des textes pour tous les mots
    all_text = ' '.join(data['Review'].dropna().astype(str).unique().tolist())
    preprocessed_all_text = preprocess_text(all_text)

    # Création des objets WordCloud
    wordcloud_positive = WordCloud(width=800, height=400, background_color='white').generate(preprocessed_positive_text)
    wordcloud_negative = WordCloud(width=800, height=400, background_color='white').generate(preprocessed_negative_text)
    wordcloud_all = WordCloud(width=800, height=400, background_color='white').generate(preprocessed_all_text)

    # Sauvegarde des nuages de mots en tant qu'images


    wordcloud_positive.to_file(wordcloud_positive_filename)
    wordcloud_negative.to_file(wordcloud_negative_filename)
    wordcloud_all.to_file(wordcloud_all_filename)

    return wordcloud_positive_filename, wordcloud_negative_filename, wordcloud_all_filename



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

    # Plot du graphique en camembert
    labels = sentiment_counts.index
    sizes = sentiment_counts.values

    colors = ['#00ff00', '#66ff66', 'gray', '#ff6666', '#ff0000']
    explode = (0, 0, 0, 0, 0)

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
