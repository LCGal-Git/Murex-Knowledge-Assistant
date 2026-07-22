#Summary of code
#Add Streamlit for chat UI

#Run installs
#python -m pip install openai python-dotenv chromadb streamlit

"""
Day 5 - Streamlit interface.

Everything about HOW retrieval and answering works is unchanged from
Day 3/4 - same chunking, same embeddings, same Chroma vector search,
same distance threshold. This file only changes HOW you interact with
it: instead of typing in a terminal, you get a browser-based chat
window.

IMPORTANT STREAMLIT CONCEPT (read this before the code):
Streamlit re-runs this entire script top-to-bottom every time you do
ANYTHING on the page (send a message, click a button). If we weren't
careful, that would mean rebuilding the vector database from scratch
on every single question - slow, and it would eventually error out
from trying to add duplicate entries.

Two Streamlit tools fix this:
1. @st.cache_resource - tells Streamlit "only actually run this
   function once, then reuse the result on every future re-run"
2. st.session_state - lets us remember the conversation history across
   re-runs, so the chat doesn't reset itself after every message

Setup:
1. pip install streamlit chromadb openai python-dotenv
2. Run with:  streamlit run day5_app.py
   (NOT "python day5_app.py" - Streamlit apps need the "streamlit run"
   command so it starts a local web server and opens your browser)
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
import streamlit as st

load_dotenv()

# CONFIG
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DOCUMENTS_FOLDER = "documents"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
NUM_CHUNKS_TO_RETRIEVE = 3
MAX_DISTANCE_THRESHOLD = 1.1

if not DEEPSEEK_API_KEY:
    st.error("DEEPSEEK_API_KEY not found. Check your .env file.")
    st.stop()  # halts the app cleanly instead of crashing with a traceback

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")


# ---- Same functions as Day 3/4, unchanged ----
def load_documents(folder_path):
    documents = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                documents[filename] = f.read()
    return documents


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def ask_llm(retrieved_chunks, question):
    context = "\n\n---\n\n".join(retrieved_chunks)

    prompt = f"""You are a helpful assistant answering questions using ONLY
the context provided below. If the answer is not in the context, say
"I don't have that information in the provided documents" - do not guess.

CONTEXT:
{context}

QUESTION:
{question}
"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )

    return response.choices[0].message.content


# ---- NEW: cached database setup ----
# @st.cache_resource means this function's code only actually executes
# ONCE (the first time the app loads), no matter how many questions get
# asked afterward. Streamlit remembers the result (the collection) and
# just hands it back on every re-run instead of rebuilding it.
@st.cache_resource
def get_vector_collection():
    chroma_client = chromadb.PersistentClient(path="chroma_db")
    collection = chroma_client.get_or_create_collection(name="murex_docs")

    # Only build the index if it's empty - this makes it safe to restart
    # the app without erroring out on duplicate entries.
    if collection.count() == 0:
        documents = load_documents(DOCUMENTS_FOLDER)
        chunk_id = 0
        for filename, text in documents.items():
            chunks = chunk_text(text)
            for chunk in chunks:
                collection.add(
                    documents=[chunk],
                    metadatas=[{"source": filename}],
                    ids=[f"chunk_{chunk_id}"]
                )
                chunk_id += 1

    return collection


def retrieve_relevant_chunks(question, collection, n_results=NUM_CHUNKS_TO_RETRIEVE):
    results = collection.query(query_texts=[question], n_results=n_results)

    retrieved_chunks = results["documents"][0]
    sources = [meta["source"] for meta in results["metadatas"][0]]
    distances = results["distances"][0]

    filtered_chunks = []
    filtered_sources = []
    for chunk, source, distance in zip(retrieved_chunks, sources, distances):
        if distance <= MAX_DISTANCE_THRESHOLD:
            filtered_chunks.append(chunk)
            filtered_sources.append(source)

    return filtered_chunks, filtered_sources


# ==================== STREAMLIT UI STARTS HERE ====================

st.title("Murex Knowledge Assistant")
st.caption("Ask a question about the indexed project documentation.")

collection = get_vector_collection()

# session_state stores the chat history so it survives across re-runs.
# Without this, every new message would wipe out everything said before.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Redraw the full conversation history on every re-run
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# st.chat_input shows a chat box pinned to the bottom of the page
question = st.chat_input("Ask a question...")

if question:
    # Show the user's own message immediately
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Retrieve + answer (identical logic to the terminal version)
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            retrieved_chunks, sources = retrieve_relevant_chunks(question, collection)

            if not retrieved_chunks:
                answer = "I don't have that information in the provided documents."
            else:
                answer = ask_llm(retrieved_chunks, question)
                unique_sources = sorted(set(sources))
                answer += f"\n\n*Source document(s): {', '.join(unique_sources)}*"

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})