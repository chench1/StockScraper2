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
        if compound > 0.05:
            return 'Positive'  # Positive sentiment
        elif compound < -0.05:
            return 'Negative'  # Negative sentiment
        else:
            return 'Neutral'  # Neutral sentiment (you can adjust this threshold)

    # Map compound scores to sentiment labels
    predicted_sentiments = [map_to_sentiment_label(compound) for compound in predicted_sentiments]

    # Combine the headlines and their predicted sentiment labels
    headline_sentiments = list(zip(headlines, predicted_sentiments))

    # Output the sentiment labels for the web-scraped headlines
    for headline, sentiment in headline_sentiments:
        print(f'Headline: {headline}\nPredicted Sentiment: {sentiment}\n')

    # Calculate the overall sentiment by considering the majority sentiment label
    sentiment_counts = Counter(predicted_sentiments)
    overall_sentiment = sentiment_counts.most_common(1)[0][0]

    # Map the overall sentiment label to an integer
    overall_sentiment_int = 0  # Default to neutral
    if overall_sentiment == 'Positive':
        overall_sentiment_int = 1
    elif overall_sentiment == 'Negative':
        overall_sentiment_int = -1

    # Calculate the certainty as a percentage
    certainty_percentage = (sentiment_counts[overall_sentiment] / len(predicted_sentiments)) * 100

    # Print the overall sentiment as an integer and the certainty percentage
    #print(f'Overall Sentiment (Integer Format): {overall_sentiment_int}')
    #print(f'Certainty of Overall Sentiment: {certainty_percentage:.2f}%')

    return [overall_sentiment, certainty_percentage]
