import os
import asyncio
import logging
from openai import OpenAI
from typing import List, Dict
from scraper import scrape_cases

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_openai_client():
    if not os.environ.get("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY not found in environment variables.")
    return OpenAI()

def extract_search_terms(question: str) -> List[str]:
    """
    Uses LLM to extract multiple relevant search phrases from the user's question.
    Returns a list of short Turkish search phrases, one per distinct concept.
    """
    openai_client = get_openai_client()

    system_prompt = (
        "You are a legal search assistant for Turkish administrative law (Danıştay). "
        "Extract the most relevant Turkish search phrases from the user's question. "
        "Return one phrase per line — each phrase should be 2-3 words max, no explanation. "
        "Return between 1 and 4 phrases depending on how many distinct legal concepts the question covers. "
        "Do not repeat the same concept twice."
    )

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            temperature=0
        )
        raw = response.choices[0].message.content.strip()
        phrases = [line.strip() for line in raw.splitlines() if line.strip()]
        logger.info(f"Extracted search phrases: {phrases}")
        return phrases
    except Exception as e:
        logger.error(f"Error extracting search terms: {e}")
        return [question]

def generate_answer(question: str, cases: List[Dict]) -> str:
    """
    Generates an answer to the user's question based on the retrieved cases.
    """
    if not cases:
        return "No relevant cases were found. Please try rephrasing your question or using different search terms."
    
    openai_client = get_openai_client()
    
    # Format cases for the prompt
    cases_text = ""
    for i, case in enumerate(cases, 1):
        cases_text += f"\n--- Case {i} ---\n"
        cases_text += f"URL: {case.get('url', 'N/A')}\n"
        cases_text += f"Content:\n{case.get('content', '')[:10000]}\n"  # Limit to 10k chars per case
    
    system_prompt = (
        "You are a helpful and knowledgeable legal assistant specializing in Turkish administrative law (Danıştay kararları). "
        "Use the following retrieved case documents to answer the user's question. "
        "Provide a comprehensive answer that references the relevant cases. "
        "If the cases don't contain sufficient information to answer the question, acknowledge this limitation. "
        "Format your answer clearly with bullet points or numbered lists where appropriate. "
        "IMPORTANT: Always respond in the same language the user's question is written in."
    )
    
    user_prompt = f"Question: {question}\n\nRetrieved Cases:\n{cases_text}"
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return f"Error generating answer: {e}"

async def agentic_search(question: str, max_cases: int = 5, phrases: List[str] = None) -> str:
    """
    Main agentic search workflow:
    1. Extract multiple search phrases from the question
    2. Scrape cases for each phrase and combine results
    3. Generate an answer based on all collected cases
    """
    logger.info(f"Processing question: '{question}'")

    # Step 1: Extract search phrases (or use pre-extracted ones from the UI)
    if phrases is None:
        phrases = extract_search_terms(question)

    # Step 2: Scrape cases for each phrase, deduplicate by URL
    all_cases: List[Dict] = []
    seen_urls: set = set()

    for phrase in phrases:
        logger.info(f"Scraping cases for phrase: '{phrase}'")
        cases = await scrape_cases(search_query=phrase, max_cases=max_cases)
        for case in cases:
            url = case.get("url", "")
            if url not in seen_urls:
                seen_urls.add(url)
                all_cases.append(case)

    if not all_cases:
        logger.warning("No cases found during scraping")
        return "No relevant cases were found on the website. Please try rephrasing your question."

    logger.info(f"Total unique cases collected: {len(all_cases)}")

    # Step 3: Generate answer from all cases
    return generate_answer(question, all_cases)

def answer_question(question: str, max_cases: int = 5, phrases: List[str] = None) -> str:
    """
    Synchronous wrapper for agentic_search.
    """
    return asyncio.run(agentic_search(question, max_cases, phrases))
