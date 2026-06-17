import streamlit as st 
import os
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from tools import web_search, scrape_url

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
# LLM Setup (Mistral)
# =========================

MISTRAL_API_KEY = get_api_key("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    raise ValueError("❌ MISTRAL_API_KEY not found! Please set it in Streamlit secrets or .env file.")

llm = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0,
    api_key=MISTRAL_API_KEY
)

# =========================
# Search Agent
# =========================

def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search]
    )

# =========================
# Reader Agent
# =========================

def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url]
    )

# =========================
# Writer Chain
# =========================

writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert research writer. Write clear, structured and insightful reports."
    ),
    (
        "human",
        """
Write a detailed research report on the topic below.

Topic:
{topic}

Research Gathered:
{research}

Structure the report as:

- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional.
"""
    ),
])

writer_chain = writer_prompt | llm | StrOutputParser()

# =========================
# Critic Chain
# =========================

critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a sharp and constructive research critic. Be honest and specific."
    ),
    (
        "human",
        """
Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
...
"""
    ),
])

critic_chain = critic_prompt | llm | StrOutputParser()

# =========================
# Example Usage (Local Testing)
# =========================

if __name__ == "__main__":
    topic = "Artificial Intelligence in Healthcare"
    sample_research = """
    AI is transforming healthcare through diagnostic assistance,
    predictive analytics, personalized treatment recommendations,
    and drug discovery.

    Sources:
    https://www.who.int
    https://www.nih.gov
    """

    report = writer_chain.invoke({
        "topic": topic,
        "research": sample_research
    })

    print("\n" + "=" * 50)
    print("RESEARCH REPORT")
    print("=" * 50)
    print(report)

    review = critic_chain.invoke({
        "report": report
    })

    print("\n" + "=" * 50)
    print("CRITIC REVIEW")
    print("=" * 50)
    print(review)