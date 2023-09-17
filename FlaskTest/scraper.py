import requests
from bs4 import BeautifulSoup

def get_news_headlines(ticker):
    page = requests.get("https://seekingalpha.com/symbol/" + ticker + "/news")
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
    
    return headlines

