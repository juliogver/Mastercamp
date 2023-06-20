import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


def create_word_cloud(csv_file):
    # Read the CSV file using pandas
    data = pd.read_csv(csv_file)

    # Concatenate the values of the 'Review' column into a single string
    text = ' '.join(data['Review'].dropna().astype(str).tolist())

    # Create the WordCloud object
    wordcloud = WordCloud(width=400, height=200,
                          background_color='white').generate(text)

    # Create a subplot for the word cloud
    plt.subplot(1, 2, 1)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud')


def create_sentiment_pie_chart(csv_file):
    # Read the CSV file using pandas
    data = pd.read_csv(csv_file)

    # Remove rows with missing values in the "Sentiment" column
    data = data.dropna(subset=['Sentiment'])

    # Count the occurrences of each sentiment category
    sentiment_counts = data['Sentiment'].value_counts()

    # Plot the pie chart
    labels = sentiment_counts.index
    sizes = sentiment_counts.values

    # Define colors for each sentiment category
    colors = ['#00ff00', '#66ff66', 'gray', '#ff6666', '#ff0000']
    explode = (0.1, 0, 0, 0, 0)  # Explode the first slice (very positive)

    # Apply CSS-like styles using wedgeprops
    wedgeprops = {'edgecolor': 'white', 'linewidth': 1}

    # Create a subplot for the pie chart
    plt.subplot(1, 2, 2)
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', startangle=90, wedgeprops=wedgeprops)
    plt.axis('equal')
    plt.title('Sentiment Proportions')


def main():
    csv_file = './System/Datas/concatenated.csv'
    model_file = './System/ia/ia_models/neural_network_model4.h5'

    create_word_cloud(csv_file)
    create_sentiment_pie_chart(csv_file)

    # Load the pre-trained neural network model
    model = load_model(model_file)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
