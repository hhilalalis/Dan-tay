import os
import asyncio
import uuid

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from agentic_search import check_scope, extract_search_terms, agentic_search

st.set_page_config(
    page_title="Danıştay Case Law Q&A",
    page_icon="⚖️",
    layout="centered"
)

# ── Session state init ────────────────────────────────────────────────────────
if "conversations" not in st.session_state:
    st.session_state.conversations = {}   # id -> {title, messages}
if "current_conv_id" not in st.session_state:
    st.session_state.current_conv_id = None

def new_conversation():
    # If already on an empty chat, just stay there (don't stack up blank chats)
    if st.session_state.current_conv_id:
        current = st.session_state.conversations.get(st.session_state.current_conv_id)
        if current and len(current["messages"]) == 0:
            return
    conv_id = str(uuid.uuid4())
    st.session_state.conversations[conv_id] = {"title": "New Chat", "messages": []}
    st.session_state.current_conv_id = conv_id

def delete_conversation(conv_id):
    del st.session_state.conversations[conv_id]
    if st.session_state.current_conv_id == conv_id:
        remaining = list(st.session_state.conversations.keys())
        if remaining:
            st.session_state.current_conv_id = remaining[-1]
        else:
            new_conversation()

# Always have at least one conversation open
if not st.session_state.current_conv_id or \
   st.session_state.current_conv_id not in st.session_state.conversations:
    new_conversation()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    if st.button("＋  New Chat", use_container_width=True, type="primary"):
        new_conversation()
        st.rerun()

    st.divider()

    # Conversation list — newest first
    for conv_id in reversed(list(st.session_state.conversations.keys())):
        conv = st.session_state.conversations[conv_id]
        label = conv["title"][:32] + ("…" if len(conv["title"]) > 32 else "")
        is_active = conv_id == st.session_state.current_conv_id
        col1, col2 = st.columns([5, 1])
        with col1:
            if st.button(label, key=f"conv_{conv_id}", use_container_width=True,
                         type="primary" if is_active else "secondary"):
                st.session_state.current_conv_id = conv_id
                st.rerun()
        with col2:
            if st.button("✕", key=f"del_{conv_id}"):
                delete_conversation(conv_id)
                st.rerun()

    st.divider()
    max_cases = st.slider("Max cases per phrase", min_value=1, max_value=10, value=3)
    st.caption(
        "Searches [karararama.danistay.gov.tr](https://karararama.danistay.gov.tr) "
        "using GPT-4.1."
    )

# ── API key guard ─────────────────────────────────────────────────────────────
if not os.environ.get("OPENAI_API_KEY"):
    st.warning("OPENAI_API_KEY is not set. Add it to your .env file or Streamlit secrets.")
    st.stop()

# ── Main chat area ────────────────────────────────────────────────────────────
conv = st.session_state.conversations[st.session_state.current_conv_id]
messages = conv["messages"]

# Show a welcome title only on empty chats
if not messages:
    st.title("⚖️ Danıştay Case Law Q&A")
    st.caption("Ask a question below to start a new conversation.")

for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a legal question in English or Turkish..."):
    # Name the conversation after the first user message
    if conv["title"] == "New Chat":
        conv["title"] = prompt[:50]

    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("Searching Danıştay...", expanded=True) as status:
            st.write("Checking if this is within Danıştay's jurisdiction...")
            scope = check_scope(prompt)

            if not scope.get("in_scope", True):
                status.update(label="Outside Danıştay's scope", state="error", expanded=False)
                response = f"⚠️ **Bu soru Danıştay'ın yetki alanı dışındadır.**\n\n{scope.get('reason', '')}"
            else:
                st.write("Extracting search terms...")
                phrases = extract_search_terms(prompt)
                st.write(f"Searching for: **{' | '.join(phrases)}**")
                st.write(f"Scraping up to {max_cases} case(s) per phrase...")
                response = asyncio.run(agentic_search(prompt, max_cases=max_cases, phrases=phrases))
                status.update(label="Done!", state="complete", expanded=False)
        st.markdown(response)

    messages.append({"role": "assistant", "content": response})
    st.rerun()  # refresh sidebar title immediately
