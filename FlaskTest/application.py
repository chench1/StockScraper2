from flask import Flask, request, render_template, send_from_directory
import logging
import pandas as pd
import pickle
import yfinance as yf
from datetime import datetime

import requests
from bs4 import BeautifulSoup

import nltk

from scraper import get_news_headlines
from scraper2 import calculate


app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))
tfvector = pickle.load(open('tfvector.pkl', 'rb'))
nltk.downloader.download('vader_lexicon')

array = [""]

@app.route('/staticFiles/<path:path>')
def send_report(path):
    print(path)
    return send_from_directory('staticFiles/', path)

@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route('/submitTicker', methods = ['POST'])
def get_ticker_prediction():
    args = request.form
    ticker = args["ticker"]

    # headlines = get_news_headlines(ticker)
    # prediction = predict(headlines).tolist()
    # prediction.append(ticker)

    overall_sentiment_int = calculate(ticker)
    
    history = get_1_yr_history(ticker)
    return render_template('index.html', data={"prediction" : overall_sentiment_int[0],  "certainty": overall_sentiment_int[1], "history" : history.to_json(orient='split')})


def predict(headlines):
    dfTest = pd.DataFrame({'Top1': headlines})
    tfidf_matrix = tfvector.transform(dfTest['Top1'])
    return model.predict(tfidf_matrix)

# params: ticker (str)
def get_1_yr_history(ticker):
    end_date = datetime.now().strftime('%Y-%m-%d')
    ticker_hist = yf.Ticker(ticker).history(interval='1d', period="6mo")
    # ticker_hist = ticker_hist.stack(level=1).rename_axis(['Date', 'Ticker']).reset_index(level=1)


    return ticker_hist.drop(columns=["Open", "High", "Low", "Volume","Dividends","Stock Splits"])



if __name__ == '__main__':
    app.debug = True
    app.run(host='127.0.0.1', port=8000)
