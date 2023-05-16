# E. Culurciello
# May 2023

# get webpages text and perform sentiment analysis on the articles

# get links from google news:
# https://stackoverflow.com/questions/1936466/how-to-scrape-only-visible-webpage-text-with-beautifulsoup

# get main text from webpage:
# https://trafilatura.readthedocs.io/en/latest/


import json
import argparse
import requests
from bs4 import BeautifulSoup
import trafilatura

from nltk import tokenize
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline


title = '>>> a sentimental scraper <<<'

def get_args():
    parser = argparse.ArgumentParser(description=title)
    arg = parser.add_argument
    arg('--i', type=str, default="nvidia",  help='search text')
    args = parser.parse_args()
    return args

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
sentiment_analyzer = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)

headers = {
        "User-Agent":
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.54 Safari/537.36"
    }


def getNewsData(query, num_results=10):
    search_req = "https://www.google.com/search?q="+query+"&gl=us&tbm=nws&num="+str(num_results)+""
    print(bcolors.OKGREEN + "ANALYZING:", search_req, "..."+bcolors.ENDC)
    response = requests.get(search_req, headers=headers)
    news_results = []

    # get webpage 
    soup = BeautifulSoup(response.content, "html.parser")

    for el in soup.select("div.SoaBEf"):
        sublink = el.find("a")["href"]
        downloaded = trafilatura.fetch_url(sublink)
        html_text = trafilatura.extract(downloaded)
        if html_text:
            sentences = tokenize.sent_tokenize(html_text)
            sentiment = sentiment_analyzer(sentences)
            sum = 0
            neutrals = 0
            if len(sentiment) > 0:
                for r in sentiment: 
                    sum += (r["label"] == "Positive")
                    neutrals += (r["label"] == "Neutral")

                den = len(sentiment)-neutrals
                sentiment = sum/den if den > 0 else 1.0 # as all neutral

                news_results.append(
                    {
                        "link": el.find("a")["href"],
                        "title": el.select_one("div.MBeuO").get_text(),
                        "snippet": el.select_one(".GI74Re").get_text(),
                        "date": el.select_one(".LfVVr").get_text(),
                        "source": el.select_one(".NUnG9d span").get_text(),
                        # "text": text,
                        "sentiment": sentiment,
                    }
                )

    return news_results, search_req


if __name__ == "__main__":
    args = get_args() # all input arguments
    print(bcolors.HEADER + title + bcolors.ENDC)
    news_results, search_req = getNewsData(args.i, num_results=10)
    print("You searched for:", args.i, "with:", search_req)
    print(bcolors.OKGREEN + "News Results:" + bcolors.ENDC)
    print(json.dumps(news_results, indent=2))