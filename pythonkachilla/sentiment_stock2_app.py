import streamlit as st
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Function to fetch and extract text from a URL
def fetch_article_text(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for request errors
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching the URL: {e}")
        return None
    
    soup = BeautifulSoup(response.content, 'html.parser')
    paragraphs = soup.find_all(['p', 'div'])  # Include <div> tags
    article_text = ' '.join([p.get_text() for p in paragraphs])
    return article_text

# Function to get the stock name from the symbol
def get_stock_name(symbol):
    try:
        stock = yf.Ticker(symbol)
        return stock.info['longName']
    except KeyError:
        st.error(f"Error fetching stock information for symbol: {symbol}")
        return None

# Function to filter text relevant to the stock
def filter_relevant_text(article_text, stock_name, stock_symbol):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(article_text)
    sentences = list(doc.sents)  # Convert the generator to a list
    
    relevant_text = []
    for i, sent in enumerate(sentences):
        if stock_name in sent.text or stock_symbol in sent.text:
            relevant_text.append(sent.text)
            if i > 0:  # Add previous sentence
                relevant_text.append(sentences[i-1].text)
            if i < len(sentences) - 1:  # Add next sentence
                relevant_text.append(sentences[i+1].text)

    return ' '.join(relevant_text)

# Function to analyze sentiment using VADER
def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_dict = analyzer.polarity_scores(text)
    sentiment_score = sentiment_dict['compound']
    return sentiment_score, sentiment_dict

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
    if not article_text:
        return None, None
    
    # Get stock name
    stock_name = get_stock_name(symbol)
    if not stock_name:
        return None, None
    
    # Filter relevant text
    relevant_text = filter_relevant_text(article_text, stock_name, symbol)
    
    # Analyze sentiment
    sentiment_score, sentiment_dict = analyze_sentiment(relevant_text)
    
    # Classify sentiment
    sentiment_class = classify_sentiment(sentiment_score)
    
    # Return sentiment score and class
    return sentiment_score, sentiment_class, sentiment_dict

# Streamlit app interface
st.title("Stock Sentiment Analysis")

symbol = st.text_input("Enter Stock Symbol (e.g., NVDA):")
url = st.text_input("Enter Article URL:")

if st.button("Get Sentiment Score"):
    if symbol and url:
        sentiment_score, sentiment_class, sentiment_dict = get_stock_sentiment(symbol, url)
        if sentiment_score is not None:
            st.write(f"The sentiment score for {symbol} is: {sentiment_score}, which is classified as: {sentiment_class}")
            st.write(f"Detailed sentiment scores: {sentiment_dict}")
        else:
            st.write("An error occurred while fetching the sentiment score.")
    else:
        st.write("Please enter both a stock symbol and a URL.")
