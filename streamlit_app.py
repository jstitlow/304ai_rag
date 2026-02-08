# Complete Streamlit RAG App with persistent data folder + streaming + avatars

import streamlit as st
import os
import subprocess
from utils.embedding import get_embedding, chunk_text
from utils.init_db import init_db, add_document, retrieve_similar, debug_db_stats
from utils.loader import load_document
from utils.llm import query_llm_stream

# ----------------------------
# Config
# ----------------------------
DATA_FOLDER = "data"
LOGO = "./Images/304ai_logo.ico"
os.makedirs(DATA_FOLDER, exist_ok=True)

# ----------------------------
# Helper: List Ollama models via CLI
# ----------------------------
def get_installed_ollama_models():
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split("\n")
        model_lines = [line for line in lines if line and not line.startswith("NAME")]
        return [line.split()[0] for line in model_lines]
    except Exception as e:
        st.sidebar.error(f"Error fetching Ollama models: {e}")
        return []

# ----------------------------
# Initialize DB
# ----------------------------
init_db()

# ----------------------------
# Sidebar
# ----------------------------
st.title("304ai DATA ASSISTANT")
st.sidebar.header("Settings")
all_models = get_installed_ollama_models()

selected_llm = st.sidebar.selectbox("LLM Model", all_models)

default_embedding_model = "nomic-embed-text:latest"
default_index = all_models.index(default_embedding_model) if default_embedding_model in all_models else 0
selected_embedding_model = st.sidebar.selectbox("Embedding Model", all_models, index=default_index)

prompt_template = st.sidebar.text_area(
    "Prompt Template",
    """You are a helpful assistant answering questions using ONLY the provided context.
If the answer is not contained in the context, say: \"I could not find that information in the documents.\"
Context:
{context}

Question: {question}
Answer:""",
    height=220
)

if st.sidebar.button("DB Status"):
    count, dims = debug_db_stats()
    st.sidebar.write(f"Chunks stored: {count}")
    st.sidebar.write(f"Embedding dimensions: {dims}")

# ----------------------------
# Ingest new uploads
# ----------------------------
uploaded_file = st.file_uploader("Upload document", type=["txt", "pdf"], label_visibility="collapsed")

if uploaded_file:
    save_path = os.path.join(DATA_FOLDER, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Saved {uploaded_file.name}")

# ----------------------------
# Index all files in data folder
# ----------------------------
@st.cache_data(show_spinner=True)
def ingest_folder(folder, embedding_model):
    indexed = 0
    for fname in os.listdir(folder):
        fpath = os.path.join(folder, fname)
        if not os.path.isfile(fpath):
            continue
        text = load_document(fpath)
        chunks = chunk_text(text)
        for chunk in chunks:
            embedding = get_embedding(chunk, model=embedding_model)
            add_document(chunk, embedding)
            indexed += 1
    return indexed

st.sidebar.button("Reindex Documents", on_click=lambda: ingest_folder(DATA_FOLDER, selected_embedding_model))

# ----------------------------
# Chat Session State
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=msg.get("avatar")):
        st.markdown(msg["content"])

# ----------------------------
# Chat Input
# ----------------------------
if user_input := st.chat_input("Ask a question…"):
    # User message
    st.session_state.messages.append({"role": "user", "content": user_input, "avatar": "🦖"})
    with st.chat_message("user", avatar="🦖"):
        st.markdown(user_input)

    # Retrieve context
    query_vec = get_embedding(user_input, model=selected_embedding_model)
    docs = retrieve_similar(query_vec, top_k=5)
    context = "\n\n".join([d[0] for d in docs])

    if context.strip():
        formatted_prompt = prompt_template.format(context=context, question=user_input)
    else:
        formatted_prompt = user_input

    # Streaming assistant response
    with st.chat_message("assistant", avatar=LOGO):
        placeholder = st.empty()
        full_response = ""
        for chunk in query_llm_stream(formatted_prompt, model=selected_llm):
            full_response += chunk
            placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response, "avatar": LOGO})