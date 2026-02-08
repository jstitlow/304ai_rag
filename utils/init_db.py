import os
import hashlib
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
# DETERMINE VECTOR SIZE
# -----------------------------
VECTOR_DIM = len(get_embedding("dimension test"))

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
    """Create documents table with filename, file_hash, content, embedding"""
    ensure_database()
    conn = get_conn()
    cur = conn.cursor()
    try:
        conn.autocommit = True
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # Check if table exists
        cur.execute("SELECT to_regclass('public.documents');")
        if not cur.fetchone()[0]:
            cur.execute(f"""
                CREATE TABLE documents (
                    id SERIAL PRIMARY KEY,
                    filename TEXT,
                    file_hash TEXT,
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
            print(f"🔁 Created documents table with vector dim {VECTOR_DIM}")
        else:
            # Ensure filename and file_hash exist (ALTER TABLE if needed)
            cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='documents';")
            existing_cols = [row[0] for row in cur.fetchall()]
            if "filename" not in existing_cols:
                cur.execute("ALTER TABLE documents ADD COLUMN filename TEXT;")
            if "file_hash" not in existing_cols:
                cur.execute("ALTER TABLE documents ADD COLUMN file_hash TEXT;")
            print("✅ Documents table ready (columns verified)")
    finally:
        cur.close()
        conn.close()

# -----------------------------
# ADD SINGLE DOCUMENT
# -----------------------------
def add_document(content, embedding, filename=None, file_hash=None):
    """Insert a single document chunk"""
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO documents (filename, file_hash, content, embedding) VALUES (%s, %s, %s, %s);",
            (filename, file_hash, content, embedding)
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
        return cur.fetchall()
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
# FILE HASH
# -----------------------------
def hash_file(file_path):
    """Compute SHA256 hash of a file"""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

# -----------------------------
# INGEST FOLDER (INCREMENTAL)
# -----------------------------
def ingest_folder(embedding_model):
    """
    Incrementally index all files in /data.
    Only new or modified files are processed.
    Returns total new chunks inserted.
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

            file_hash = hash_file(fpath)

            # Check if file is already indexed with same hash
            cur.execute("""
                SELECT 1 FROM documents
                WHERE filename = %s AND file_hash = %s
                LIMIT 1;
            """, (fname, file_hash))
            if cur.fetchone():
                continue  # already indexed

            # Load and chunk file
            try:
                text = load_document(fpath)
            except Exception as e:
                print(f"⚠️ Skipping {fpath}: {e}")
                continue

            chunks = chunk_text(text)
            for chunk in chunks:
                emb = get_embedding(chunk, model=embedding_model)
                cur.execute("""
                    INSERT INTO documents (filename, file_hash, content, embedding)
                    VALUES (%s, %s, %s, %s);
                """, (fname, file_hash, chunk, emb))
                total_chunks += 1

        conn.commit()
        return total_chunks
    finally:
        cur.close()
        conn.close()
