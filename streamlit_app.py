import streamlit as st
import os

from utils.embedding import get_embedding, chunk_text
from utils.init_db import init_db, retrieve_similar, debug_db_stats, clear_documents, ingest_folder, add_document, hash_file
from utils.llm import query_llm_stream
from utils.loader import load_document
from utils.llm import get_installed_ollama_models

# ----------------------------
# Config
# ----------------------------
DATA_FOLDER = "data"
os.makedirs(DATA_FOLDER, exist_ok=True)
LOGO = "./Images/304ai_logo.ico"

# ----------------------------
# Initialize DB & load ollam models
# ----------------------------
init_db()
get_installed_ollama_models()

# ----------------------------
# Sidebar
# ----------------------------
st.title("304ai Data Assistant")
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
    st.sidebar.write(f"Embedding dimensions: {dims:.0f}" if dims else "Embedding dimensions: N/A")

if st.sidebar.button("Reindex Documents"):
    st.sidebar.info("Reindexing... this may take a few seconds.")
    #clear_documents()
    total_chunks = ingest_folder(selected_embedding_model)
    st.sidebar.success(f"Reindex complete — {total_chunks} chunks stored.")


# ----------------------------
# Ingest new uploads
# ----------------------------
uploaded_file = st.file_uploader(
    "Upload document", type=["txt", "pdf"], label_visibility="collapsed"
)

if uploaded_file:
    save_path = os.path.join(DATA_FOLDER, uploaded_file.name)

    if os.path.exists(save_path):
        st.info(f"File '{uploaded_file.name}' is already uploaded.")
    else:
        # Save file to data folder
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Saved '{uploaded_file.name}'")

        # ----------------------------
        # Embed and add to database
        # ----------------------------
        try:
            # Load text from the uploaded file
            text = load_document(save_path)

            # Split into chunks
            chunks = chunk_text(text)

            # Track how many chunks were added
            total_chunks = 0
            for chunk in chunks:
                embedding = get_embedding(chunk, model=selected_embedding_model)
                add_document(
                    content=chunk,
                    embedding=embedding,
                    filename=uploaded_file.name,
                    file_hash=hash_file(save_path)
                )
                total_chunks += 1

            st.success(f"Indexed {total_chunks} chunks from '{uploaded_file.name}'")
        except Exception as e:
            st.error(f"⚠️ Failed to process '{uploaded_file.name}': {e}")

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