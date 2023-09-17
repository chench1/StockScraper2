import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter

def calculate(ticker):
    # Load your labeled data from the CSV file
    df = pd.read_csv('all-data.csv', encoding="ISO-8859-1")  # Replace with your file path

    # Initialize the SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()

    # Web scraping part
    url = "https://seekingalpha.com/symbol/" + ticker + "/news"
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    headlines = []

    marker = False

    for link in soup.find_all('a'):
        if not marker:
            marker = link.get_text().strip() == "Related Analysis"
        else:
            headline = link.get_text().split()
            if len(headline) > 2:
                headlines.append(" ".join(headline))

    # Predict sentiments and store them in a new column 'predicted_sentiment'
    predicted_sentiments = [sia.polarity_scores(text)['compound'] for text in headlines]

    # Define a function to map the compound scores to sentiment labels
    def map_to_sentiment_label(compound):
        if compound > 0.1:  # Adjust this threshold for a more positive sentiment
            return 'Positive'  # Positive sentiment
        elif compound < -0.1:  # Adjust this threshold for a more negative sentiment
            return 'Negative'  # Negative sentiment
        else:
            return None  # Exclude 'Neutral' sentiment
        
    predicted_sentiments = [map_to_sentiment_label(compound) for compound in predicted_sentiments if map_to_sentiment_label(compound) is not None]

    # Calculate the overall sentiment by considering the majority sentiment label
    sentiment_counts = Counter(predicted_sentiments)

    if sentiment_counts['Positive'] > sentiment_counts['Negative']:
        overall_sentiment_str = 'Positive'
    else:
        overall_sentiment_str = 'Negative'

    # Calculate the certainty as a percentage
    certainty_percentage = (sentiment_counts[overall_sentiment_str] / len(predicted_sentiments)) * 100

    # Print the overall sentiment and the certainty percentage
    return[overall_sentiment_str, certainty_percentage]