from fastapi import FastAPI, HTTPException
from bs4 import BeautifulSoup
import requests

app = FastAPI()



def scrape_website(url: str):
    try:
        # Fetch the HTML content of the website
        response = requests.get(url)
        if response.status_code != 200:
            raise HTTPException(status_code=404, detail="Website not found")

        # Parse the HTML using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract desired data from the website
        # For example, let's extract all the links
        links = [link.get('href') for link in soup.find_all('a') if link.get('href')]

        return links

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Web Scraping API!"}

@app.get("/scrape/")
async def scrape_website_api(url: str):
    if not url:
        raise HTTPException(status_code=400, detail="URL parameter is required")
    links = scrape_website(url)
    return {"links": links}

