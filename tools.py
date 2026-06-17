import streamlit as st  # <-- SABSE PEHLE IMPORT
import os
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
from langchain.tools import tool

# Load environment variables (Local ke liye)
load_dotenv()

# =========================
# API KEY HELPER FUNCTION
# =========================

def get_api_key(key_name: str) -> str:
    try:
        if hasattr(st, 'secrets') and key_name in st.secrets:
            return st.secrets[key_name]
    except:
        pass
    return os.getenv(key_name)

# =========================
# Tavily Setup
# =========================

TAVILY_API_KEY = get_api_key("TAVILY_API_KEY")

if not TAVILY_API_KEY:
    raise ValueError("❌ TAVILY_API_KEY not found! Please set it in Streamlit secrets or .env file.")

tavily = TavilyClient(api_key=TAVILY_API_KEY)

# =========================
# Tools
# =========================

@tool
def web_search(query: str) -> str:
    """Search the web for recent and reliable information on a topic. Returns Titles, URLs and snippets."""
    try:
        results = tavily.search(query=query, max_results=5)
        
        out = []
        for r in results['results']:
            out.append(
                f"Title: {r['title']}\nURL: {r['url']}\nSnippet: {r['content'][:300]}\n"
            )
        
        return "\n----\n".join(out) if out else "No results found."
    
    except Exception as e:
        return f"Search error: {str(e)}"

@tool
def scrape_url(url: str) -> str:
    """Scrape and return clean text content from a given URL for deeper reading."""
    try:
        resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, "html.parser")
        
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        
        text = soup.get_text(separator=" ", strip=True)[:3000]
        return text if text else "No content found on the page."
    
    except requests.exceptions.Timeout:
        return "Error: Request timed out while scraping URL."
    except requests.exceptions.RequestException as e:
        return f"Error fetching URL: {str(e)}"
    except Exception as e:
        return f"Could not scrape URL: {str(e)}"