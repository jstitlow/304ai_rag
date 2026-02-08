import os
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from utils.embedding import get_embedding
from utils.loader import load_document
from utils.embedding import chunk_text

USER = "joshtitlow"
DB_NAME = "rag_db"
PORT = 5432
HOST = "localhost"
DATA_DIR = "data"


# -----------------------------
# DETERMINE VECTOR SIZE FROM MODEL
# -----------------------------
def get_vector_dim():
    test_vec = get_embedding("dimension test")
    return len(test_vec)


VECTOR_DIM = get_vector_dim()


# -----------------------------
# DATABASE CREATION
# -----------------------------
def ensure_database():
    """Create database if it doesn't exist"""
    conn = psycopg2.connect(host=HOST, user=USER, dbname="postgres", port=PORT)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    try:
        cur.execute("SELECT 1 FROM pg_database WHERE datname=%s;", (DB_NAME,))
        if not cur.fetchone():
            cur.execute(f'CREATE DATABASE {DB_NAME};')
            print(f"🆕 Created database {DB_NAME}")
    finally:
        cur.close()
        conn.close()


# -----------------------------
# CONNECT
# -----------------------------
def get_conn():
    """Return a new connection to the RAG database"""
    return psycopg2.connect(
        host=HOST,
        user=USER,
        dbname=DB_NAME,
        port=PORT
    )


# -----------------------------
# INIT TABLE
# -----------------------------
def init_db():
    """Initialize documents table and vector extension"""
    ensure_database()
    conn = get_conn()
    cur = conn.cursor()
    try:
        conn.autocommit = True
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # Check existing dimension
        cur.execute("""
            SELECT atttypmod
            FROM pg_attribute
            WHERE attrelid = 'documents'::regclass
            AND attname = 'embedding';
        """)
        row = cur.fetchone()

        recreate = True
        if row:
            existing_dim = row[0] - 4  # pgvector internal encoding
            if existing_dim == VECTOR_DIM:
                recreate = False

        if recreate:
            print(f"🔁 Rebuilding table for vector size {VECTOR_DIM}")
            cur.execute("DROP TABLE IF EXISTS documents;")
            cur.execute(f"""
                CREATE TABLE documents (
                    id SERIAL PRIMARY KEY,
                    content TEXT,
                    embedding VECTOR({VECTOR_DIM})
                );
            """)
            cur.execute("""
                CREATE INDEX documents_embedding_idx
                ON documents
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)
        else:
            print("✅ Documents table already exists with correct vector size")

    finally:
        cur.close()
        conn.close()


# -----------------------------
# INSERT DOCUMENT
# -----------------------------
def add_document(content, embedding):
    """Insert a single document chunk"""
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO documents (content, embedding) VALUES (%s, %s);",
            (content, embedding)
        )
        conn.commit()
    finally:
        cur.close()
        conn.close()


# -----------------------------
# RETRIEVE SIMILAR
# -----------------------------
def retrieve_similar(query_embedding, top_k=5):
    """Retrieve top_k most similar documents"""
    conn = get_conn()
    cur = conn.cursor()
    try:
        vec_str = str(list(query_embedding))
        cur.execute("""
            SELECT content, embedding <#> %s::vector AS distance
            FROM documents
            ORDER BY distance
            LIMIT %s;
        """, (vec_str, top_k))
        results = cur.fetchall()
        return results
    finally:
        cur.close()
        conn.close()


# -----------------------------
# DEBUG / STATS
# -----------------------------
def debug_db_stats():
    """Return number of chunks and vector dimensions"""
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("SELECT COUNT(*) FROM documents;")
        doc_count = cur.fetchone()[0]

        cur.execute("SELECT AVG(vector_dims(embedding)) FROM documents;")
        dims = cur.fetchone()[0]

        return doc_count, dims
    finally:
        cur.close()
        conn.close()


def clear_documents():
    """Delete all documents"""
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("TRUNCATE documents;")
        conn.commit()
    finally:
        cur.close()
        conn.close()


# -----------------------------
# INGEST FOLDER
# -----------------------------
def ingest_folder(embedding_model):
    """
    Index all files in the data folder into the DB
    Returns total chunks inserted
    """
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    total_chunks = 0
    conn = get_conn()
    cur = conn.cursor()
    try:
        for fname in os.listdir(DATA_DIR):
            fpath = os.path.join(DATA_DIR, fname)
            if not os.path.isfile(fpath):
                continue
            try:
                text = load_document(fpath)
            except Exception as e:
                print(f"⚠️ Skipping {fpath}: {e}")
                continue

            chunks = chunk_text(text)
            for chunk in chunks:
                emb = get_embedding(chunk, model=embedding_model)
                cur.execute(
                    "INSERT INTO documents (content, embedding) VALUES (%s, %s);",
                    (chunk, emb)
                )
                total_chunks += 1

        conn.commit()
        return total_chunks
    finally:
        cur.close()
        conn.close()
