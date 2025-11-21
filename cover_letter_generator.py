# cover_letter_generator.py
import requests
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from bs4 import BeautifulSoup

def fetch_company_summary(company_url: str) -> str:
    """
    Scrape the company's About section or meta description for context.
    Returns a short summary string.
    """
    try:
        response = requests.get(company_url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Try to get the meta description
        meta = soup.find("meta", attrs={"name": "description"})
        if meta and meta.get("content"):
            return meta["content"]

        # Try to find an 'About' section
        about = soup.find(string=lambda t: t and "about" in t.lower())
        if about:
            return about.strip()

        # If nothing found, fallback
        return "No detailed company info available."
    except Exception as e:
        return f"Error fetching company summary: {e}"

from langchain_google_genai import ChatGoogleGenerativeAI
# Corrected model name: use 'gemini-2.5-flash' for general availability
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)


def generate_personalized_cover_letter(company_name, company_summary, job_description, resume_text):
    prompt = f"""
You are a professional career assistant.

Write a personalized cover letter for the company "{company_name}".
Use this company description:
{company_summary}

Here is the job description:
{job_description}

Base the letter on this resume content:
{resume_text}

Requirements:
- Professional, concise, and relevant to the role.
- Highlight 2–3 most relevant achievements or skills.
- Keep within 250–300 words.
    """
    response = llm.invoke(prompt)
    return response.content
