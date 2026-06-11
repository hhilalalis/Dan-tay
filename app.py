import os
import asyncio

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from agentic_search import extract_search_terms, agentic_search

st.set_page_config(
    page_title="Danıştay Case Law Q&A",
    page_icon="⚖️",
    layout="centered"
)

with st.sidebar:
    st.header("Settings")
    max_cases = st.slider("Max cases per search phrase", min_value=1, max_value=10, value=3)
    st.divider()
    st.markdown("**About**")
    st.caption(
        "Searches [karararama.danistay.gov.tr](https://karararama.danistay.gov.tr) "
        "and uses GPT-4.1 to answer your question based on real Turkish court decisions."
    )

if not os.environ.get("OPENAI_API_KEY"):
    st.warning("OPENAI_API_KEY is not set. Add it to your .env file or Streamlit secrets.")
    st.stop()

st.title("⚖️ Danıştay Case Law Q&A")
st.caption("Ask questions about Turkish administrative court decisions")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a legal question in English or Turkish..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("Searching Danıştay...", expanded=True) as status:
            st.write("Extracting search terms...")
            phrases = extract_search_terms(prompt)
            st.write(f"Searching for: **{' | '.join(phrases)}**")
            st.write(f"Scraping up to {max_cases} case(s) per phrase...")
            response = asyncio.run(agentic_search(prompt, max_cases=max_cases, phrases=phrases))
            status.update(label="Done!", state="complete", expanded=False)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
