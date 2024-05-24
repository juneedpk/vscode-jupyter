import streamlit as st
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import spacy
from textblob import TextBlob

# Function to fetch and extract text from a URL
def fetch_article_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    paragraphs = soup.find_all(['p', 'div'])  # Include <div> tags
    article_text = ' '.join([p.get_text() for p in paragraphs])
    return article_text

# Function to get the stock name from the symbol
def get_stock_name(symbol):
    stock = yf.Ticker(symbol)
    return stock.info['longName']

# Function to filter text relevant to the stock
def filter_relevant_text(article_text, stock_name, stock_symbol):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(article_text)
    sentences = [sent.text for sent in doc.sents if stock_name in sent.text or stock_symbol in sent.text]
    relevant_text = ' '.join(sentences)
    return relevant_text

# Function to analyze sentiment
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    return sentiment_score

# Function to classify sentiment score into categories
def classify_sentiment(score):
    if score <= -0.6:
        return "Very Negative"
    elif score <= -0.2:
        return "Negative"
    elif score < 0.2:
        return "Neutral"
    elif score < 0.6:
        return "Positive"
    else:
        return "Very Positive"

# Main function to tie everything together
def get_stock_sentiment(symbol, url):
    # Fetch article text
    article_text = fetch_article_text(url)
    
    # Get stock name
    stock_name = get_stock_name(symbol)
    
    # Filter relevant text
    relevant_text = filter_relevant_text(article_text, stock_name, symbol)
    
    # Analyze sentiment
    sentiment_score = analyze_sentiment(relevant_text)
    
    # Classify sentiment
    sentiment_class = classify_sentiment(sentiment_score)
    
    return sentiment_score, sentiment_class

# Streamlit app
st.title("Stock Sentiment Analysis")
st.markdown("<h1 style='text-align: center; font-size: 30px;'>Stock Sentiment Analysis</h1>", unsafe_allow_html=True)

symbol = st.text_input("Enter stock symbol:", value="NVDA")
url = st.text_input("Enter article URL:", value="https://www.investopedia.com/nvidia-q1-fy-2025-earnings-preview-8643286")

if st.button("Analyze Sentiment"):
    sentiment_score, sentiment_class = get_stock_sentiment(symbol, url)
    st.markdown(f"**Sentiment Score for {symbol}:** {sentiment_score}", unsafe_allow_html=True)
    st.markdown(f"**Sentiment Classification for {symbol}:** {sentiment_class}", unsafe_allow_html=True)
