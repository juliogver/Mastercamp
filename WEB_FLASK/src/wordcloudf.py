import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Read the CSV file
data = pd.read_csv('your_file.csv')

# Concatenate all text columns into a single string
text = ' '.join(data[column] for column in data.columns if data[column].dtype == 'object')

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400).generate(text)

# Display the word cloud using Matplotlib
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
