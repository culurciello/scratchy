# E. Culurciello
# May 2023

import json
import argparse
import requests
from bs4 import BeautifulSoup
from bs4.element import Comment

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


finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)

headers = {
        "User-Agent":
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.54 Safari/537.36"
    }

def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True


def text_from_html(search_req):
    response = requests.get(search_req, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)  
    return u" ".join(t.strip() for t in visible_texts)


def getNewsData(query, num_results=10):
    search_req = "https://www.google.com/search?q="+query+"&gl=us&tbm=nws&num="+str(num_results)+""
    response = requests.get(search_req, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")
    news_results = []

    for el in soup.select("div.SoaBEf"):
        text = text_from_html(el.find("a")["href"])
        sentences = tokenize.sent_tokenize(text)
        sentiment = nlp(sentences)
        sum = 0
        neutrals = 0
        for r in sentiment: 
            sum += (r["label"] == "Positive")
            neutrals +=  (r["label"] == "Neutral")

        news_results.append(
            {
                "link": el.find("a")["href"],
                "title": el.select_one("div.MBeuO").get_text(),
                "snippet": el.select_one(".GI74Re").get_text(),
                "date": el.select_one(".LfVVr").get_text(),
                "source": el.select_one(".NUnG9d span").get_text(),
                # "text": text,
                "sentiment": sum/(len(sentiment)-neutrals),
            }
        )

    return news_results, search_req


if __name__ == "__main__":
    args = get_args() # all input arguments
    print(title)
    news_results, search_req = getNewsData(args.i, num_results=10)
    print("You searched for:", args.i, "with:", search_req)
    print("News Results:")
    print(json.dumps(news_results, indent=2))