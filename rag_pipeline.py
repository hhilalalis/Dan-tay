import os
import uuid
import logging
from openai import OpenAI
import chromadb
from typing import List, Dict
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "danistay_cases"

# Load the local Turkish BERT model for embeddings
logger.info("Loading Turkish BERT embedding model...")
embedder = SentenceTransformer('emrecan/bert-base-turkish-cased-mean-nli-stsb-tr')
logger.info("Embedding model loaded.")

def get_openai_client():
    if not os.environ.get("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY not found in environment variables.")
    return OpenAI()

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Simple text chunking function from scratch.
    """
    if not text:
        return []
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
        
    return chunks

def chunk_and_embed_cases(cases: List[Dict]):
    """
    Takes a list of case dictionaries, chunks the content manually, 
    embeds them using a local Turkish BERT, and stores them in ChromaDB.
    """
    if not cases:
        logger.warning("No cases provided to chunk and embed.")
        return

    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    # Get or create collection
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    
    docs = []
    metadatas = []
    ids = []
    
    for case in cases:
        content = case["content"]
        chunks = chunk_text(content, chunk_size=1000, overlap=200)
        
        for i, chunk in enumerate(chunks):
            docs.append(chunk)
            metadatas.append({
                "url": case.get("url", ""),
                "query": case.get("query", ""),
                "chunk_index": i
            })
            ids.append(str(uuid.uuid4()))
            
    logger.info(f"Split {len(cases)} cases into {len(docs)} chunks.")
    
    # Process embeddings in batches to respect memory limits
    batch_size = 100
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        batch_metadatas = metadatas[i:i+batch_size]
        
        logger.info(f"Embedding and storing batch {i//batch_size + 1}...")
        
        # Get embeddings from local BERT
        embeddings = embedder.encode(batch_docs).tolist()
        
        # Add to ChromaDB
        collection.add(
            ids=batch_ids,
            embeddings=embeddings,
            metadatas=batch_metadatas,
            documents=batch_docs
        )
        
    logger.info(f"Successfully stored vectors in {CHROMA_DB_DIR}")

def answer_question(query: str) -> str:
    """
    Retrieves context from ChromaDB and synthesizes an answer using OpenAI.
    """
    if not os.path.exists(CHROMA_DB_DIR):
        return f"Error: Chroma DB directory {CHROMA_DB_DIR} does not exist. Please run scraping/indexing first."
        
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    try:
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        return f"Error accessing collection: {e}"
        
    openai_client = get_openai_client()
    
    # 1. Embed the user query with local BERT
    query_embedding = embedder.encode([query])[0].tolist()
    
    # 2. Retrieve top-k documents from Chroma
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )
    
    documents = results.get("documents", [[]])[0]
    if not documents:
        return "I could not find any relevant information in the database."
        
    context = "\n\n".join(documents)
    
    # 3. Formulate the prompt for the LLM
    system_prompt = (
        "You are a helpful and knowledgeable legal assistant specializing in Turkish administrative law (Danıştay kararları). "
        "Use the following pieces of retrieved historical case context to answer the user's question. "
        "If you don't know the answer or if the context doesn't contain the answer, just say that you don't know. "
        "Keep the answer concise and reference the context provided."
    )
    
    user_prompt = f"Context:\n{context}\n\nQuestion: {query}"
    
    # 4. Generate answer using OpenAI Chat Completion
    try:
        chat_response = openai_client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0
        )
        return chat_response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {e}")
        return str(e)
