import os
import fitz  # PyMuPDF for handling PDF files
import streamlit as st
import subprocess
import numpy as np
import faiss  # for approximate nearest neighbor search
import pickle  # for caching embeddings and queries
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import wikipedia
from collections import deque
import re
import time

# Initialize the embedder and LLM models
embedder = SentenceTransformer("Alibaba-NLP/gte-base-en-v1.5", trust_remote_code=True)
gpt2_model = GPT2LMHeadModel.from_pretrained("distilgpt2")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")

if gpt2_tokenizer.pad_token is None:
    gpt2_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    gpt2_model.resize_token_embeddings(len(gpt2_tokenizer))

# Paths to content directories and cache files
CONTENT_DIR = "Content"
MEDICAL_BOOKS_DIR = os.path.join(CONTENT_DIR, "Medical_Books")
MEDICAL_ARTICLES_DIR = os.path.join(CONTENT_DIR, "medical_articles")
EMBEDDINGS_CACHE_PATH = "embeddings_cache.pkl"
DOCUMENTS_CACHE_PATH = "documents_cache.pkl"
QUERY_CACHE_PATH = "query_cache.pkl"
FEEDBACK_CACHE_PATH = "feedback_cache.pkl"

# Load cached query-response pairs if available
def load_query_cache():
    if os.path.exists(QUERY_CACHE_PATH):
        with open(QUERY_CACHE_PATH, 'rb') as f:
            return pickle.load(f)
    return {}

# Save cached query-response pairs to disk
def save_query_cache(query_cache):
    with open(QUERY_CACHE_PATH, 'wb') as f:
        pickle.dump(query_cache, f)

# Load cached feedback if available
def load_feedback_cache():
    if os.path.exists(FEEDBACK_CACHE_PATH):
        with open(FEEDBACK_CACHE_PATH, 'rb') as f:
            return pickle.load(f)
    return {"positive": 0, "negative": 0}

# Save feedback to disk
def save_feedback_cache(feedback_cache):
    with open(FEEDBACK_CACHE_PATH, 'wb') as f:
        pickle.dump(feedback_cache, f)

# Clear cached queries and feedback
def clear_cache():
    if os.path.exists(QUERY_CACHE_PATH):
        os.remove(QUERY_CACHE_PATH)
    if os.path.exists(FEEDBACK_CACHE_PATH):
        os.remove(FEEDBACK_CACHE_PATH)
    st.session_state.query_cache = {}
    st.session_state.response_feedback = {"positive": 0, "negative": 0}
    st.session_state.response_times = []
    st.session_state.context = deque(maxlen=3)
    st.success("Cache cleared.")

# Load and cache local documents
def load_local_documents(directory):
    documents = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                if file.lower().endswith(".pdf"):
                    with fitz.open(file_path) as pdf_doc:
                        pdf_text = ""
                        for page in pdf_doc:
                            pdf_text += page.get_text()
                        documents.append(pdf_text)
                elif file.lower().endswith(".txt"):
                    with open(file_path, 'r', encoding="utf-8") as f:
                        documents.append(f.read())
            except Exception as e:
                st.warning(f"Failed to load document {file_path}: {e}")
    return documents

# Load cached embeddings if available
def load_cached_data():
    if os.path.exists(EMBEDDINGS_CACHE_PATH) and os.path.exists(DOCUMENTS_CACHE_PATH):
        with open(EMBEDDINGS_CACHE_PATH, 'rb') as f_embed, open(DOCUMENTS_CACHE_PATH, 'rb') as f_docs:
            document_embeddings = pickle.load(f_embed)
            all_documents = pickle.load(f_docs)
        st.write("Loaded embeddings and documents from cache.")
        return all_documents, document_embeddings
    else:
        return None, None

# Save embeddings and documents to cache
def save_cache_data(documents, embeddings):
    with open(EMBEDDINGS_CACHE_PATH, 'wb') as f_embed, open(DOCUMENTS_CACHE_PATH, 'wb') as f_docs:
        pickle.dump(embeddings, f_embed)
        pickle.dump(documents, f_docs)
    st.write("Saved embeddings and documents to cache.")

# Generate embeddings for local documents
def generate_embeddings_for_local_docs():
    books = load_local_documents(MEDICAL_BOOKS_DIR)
    articles = load_local_documents(MEDICAL_ARTICLES_DIR)
    all_documents = books + articles
    if not all_documents:
        st.warning("No documents found in Medical_Books or medical_articles directories.")
        return [], np.array([])

    document_embeddings = embedder.encode(all_documents, normalize_embeddings=True)
    save_cache_data(all_documents, document_embeddings)
    return all_documents, document_embeddings

# Load cached data if available, else generate new embeddings
all_documents, document_embeddings = load_cached_data()
if all_documents is None or document_embeddings is None:
    all_documents, document_embeddings = generate_embeddings_for_local_docs()

# Create a FAISS index for fast retrieval of relevant documents
def create_faiss_index(embeddings):
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index

# Set up FAISS index
faiss_index = create_faiss_index(document_embeddings)

# Cached query-responses and feedback
query_cache = load_query_cache()
feedback_cache = load_feedback_cache()

# Fetch relevant content from local documents using FAISS
def fetch_relevant_local_content(query, top_k=3, similarity_threshold=0.8):
    if query in query_cache:
        return query_cache[query]  # Return cached response if available

    if document_embeddings.size == 0:
        return "No local documents are available for retrieval."

    query_embed = embedder.encode([query], normalize_embeddings=True)
    distances, indices = faiss_index.search(query_embed, top_k)

    # Filter documents based on similarity threshold
    relevant_docs = [all_documents[idx] for i, idx in enumerate(indices[0]) if distances[0][i] < similarity_threshold]
    response = "\n\n".join(relevant_docs) if relevant_docs else "No relevant content found in local sources."

    # Cache the response for this query
    query_cache[query] = response
    save_query_cache(query_cache)  # Persist cache to disk
    return response

# Fetch relevant content from Wikipedia
def get_relevant_wiki_content(query):
    try:
        search_results = wikipedia.search(query, results=1)
        if not search_results:
            return "No relevant Wikipedia content found."
        page = wikipedia.page(search_results[0])
        paragraphs = page.content.split('\n\n')
        relevant_content = [para for para in paragraphs if re.search(r"\b" + query.split()[0] + r"\b", para, re.IGNORECASE)]
        return "\n\n".join(relevant_content[:3]) if relevant_content else "No highly relevant Wikipedia content found."
    except wikipedia.exceptions.DisambiguationError:
        return "Disambiguation error: Multiple topics found."
    except wikipedia.exceptions.PageError:
        return "The relevant Wikipedia page does not exist."
    except Exception as e:
        return f"An error occurred: {e}"

# Combine relevant content from local documents and Wikipedia
def get_relevant_content(query):
    local_content = fetch_relevant_local_content(query)
    wikipedia_content = get_relevant_wiki_content(query)

    if "No relevant" in local_content and "No relevant" in wikipedia_content:
        return "No specific information on this topic. Try rephrasing your query."
    
    return f"{local_content}\n\n{wikipedia_content}"

# Generate response using GPT-2
def call_gpt2(prompt, max_tokens=100):
    inputs = gpt2_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    output = gpt2_model.generate(
        inputs['input_ids'], 
        max_new_tokens=max_tokens, 
        pad_token_id=gpt2_tokenizer.pad_token_id, 
        do_sample=True, 
        top_p=0.85,  # Lower top-p for more focused results
        temperature=0.7  # Lower temperature for less randomness
    )
    return gpt2_tokenizer.decode(output[0], skip_special_tokens=True).strip()

# Generate response using LLaMA
def call_llama(prompt, max_tokens=100):
    try:
        process = subprocess.run(
            ['ollama', 'run', 'llama3.2:latest'],
            input=prompt,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        if process.returncode != 0:
            return f"Error running LLaMA: {process.stderr.strip()}"
        return process.stdout.strip()
    except Exception as e:
        return f"Subprocess error: {e}"

# Generate a response combining RAG and LLM
def generate_rag_response(query, use_gpt2=True):
    context_history = "\n\n".join(st.session_state.context)
    context = get_relevant_content(query)
    prompt = f"""
    Conversation history:
    {context_history}

    Based on information from reliable medical sources:
    
    {context}
    
    Answer the following question as accurately as possible:
    {query}
    """
    response = call_gpt2(prompt) if use_gpt2 else call_llama(prompt)
    st.session_state.context.append(f"User: {query}\nBot: {response}")
    return response

### --- Streamlit UI Setup --- ###

st.title("Medical RAG Chatbot")
st.write("Ask questions and choose between using RAG mode or a standalone LLM.")

# Choose mode and LLM model
mode = st.radio("Choose mode:", ["RAG (Local Content + Wikipedia + LLM)", "LLM Only"])
llm_model = st.selectbox("Choose LLM model:", ["GPT-2", "LLaMA"])

# Conversation history and response feedback
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = deque(maxlen=10)
if "response_feedback" not in st.session_state:
    st.session_state.response_feedback = feedback_cache
if "response_times" not in st.session_state:
    st.session_state.response_times = []
if "context" not in st.session_state:
    st.session_state.context = deque(maxlen=3)

# User input
user_query = st.text_input("Enter your question:")

# Generate a response when user submits a query
if st.button("Send"):
    if user_query:
        use_gpt2 = llm_model == "GPT-2"
        
        # Start timer
        start_time = time.time()
        
        # Generate response
        response = generate_rag_response(user_query, use_gpt2) if mode.startswith("RAG") else (call_gpt2(user_query) if use_gpt2 else call_llama(user_query))
        
        # Calculate and store response time
        response_time = time.time() - start_time
        st.session_state.response_times.append(response_time)

        # Append user query and bot response to conversation history
        st.session_state.conversation_history.append(f"User: {user_query}\nBot: {response}")
        st.subheader("Response:")
        st.write(response)

        # Record feedback options
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("👍", key=f"thumbs_up_{user_query}"):
                st.session_state.response_feedback["positive"] += 1
                save_feedback_cache(st.session_state.response_feedback)
        with col2:
            if st.button("👎", key=f"thumbs_down_{user_query}"):
                st.session_state.response_feedback["negative"] += 1
                save_feedback_cache(st.session_state.response_feedback)

# Display conversation history
st.subheader("Conversation History")
for message in st.session_state.conversation_history:
    st.write(message)

# Statistics and Cached Data in Sidebar
st.sidebar.header("Statistics")
st.sidebar.write(f"Total Queries: {len(st.session_state.conversation_history)}")
st.sidebar.write(f"Average Response Time: {np.mean(st.session_state.response_times):.2f} seconds" if st.session_state.response_times else "No response yet.")
st.sidebar.write(f"Current Response Time: {st.session_state.response_times[-1]:.2f} seconds" if st.session_state.response_times else "No response yet.")
st.sidebar.write(f"Total Positive Responses: {st.session_state.response_feedback['positive']}")
st.sidebar.write(f"Total Negative Responses: {st.session_state.response_feedback['negative']}")

st.sidebar.header("Cached Queries")
st.sidebar.write(list(query_cache.keys()))

# Clear Cache Button
if st.sidebar.button("Clear Cache"):
    clear_cache()
