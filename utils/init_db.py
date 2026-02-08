import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from utils.embedding import get_embedding

USER = "joshtitlow"
DB_NAME = "rag_db"
PORT = 5432
HOST = "localhost"

# -------- DETERMINE VECTOR SIZE FROM MODEL --------
def get_vector_dim():
    test_vec = get_embedding("dimension test")
    return len(test_vec)

VECTOR_DIM = get_vector_dim()


# -------- DATABASE CREATION --------
def ensure_database():
    conn = psycopg2.connect(
        host=HOST,
        user=USER,
        dbname="postgres",
        port=PORT
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()

    cur.execute("SELECT 1 FROM pg_database WHERE datname=%s;", (DB_NAME,))
    if not cur.fetchone():
        cur.execute(f'CREATE DATABASE {DB_NAME};')
        print("🆕 Created database rag_db")

    cur.close()
    conn.close()


# -------- CONNECT --------
def get_conn():
    return psycopg2.connect(
        host=HOST,
        user=USER,
        dbname=DB_NAME,
        port=PORT
    )


# -------- INIT TABLE --------
def init_db():
    ensure_database()

    conn = get_conn()
    conn.autocommit = True
    cur = conn.cursor()

    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # check existing dimension
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

    cur.close()
    conn.close()
    print(f"✅ Database ready (dim={VECTOR_DIM})")


# -------- INSERT --------
def add_document(content, embedding):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
        (content, embedding)
    )

    conn.commit()
    cur.close()
    conn.close()


# -------- RETRIEVE --------
def retrieve_similar(query_embedding, top_k=5):
    """
    Retrieve the top_k most similar documents to the query embedding.

    Args:
        query_embedding (list or numpy array of floats): The query embedding vector.
        top_k (int): Number of results to return.

    Returns:
        List of tuples: (content, distance)
    """
    conn = get_conn()
    cur = conn.cursor()

    # Convert Python list/array to a PostgreSQL vector string
    vec_str = str(list(query_embedding))  # ensures it's a plain Python list

    cur.execute("""
        SELECT content, embedding <#> %s::vector AS distance
        FROM documents
        ORDER BY distance
        LIMIT %s
    """, (vec_str, top_k))

    results = cur.fetchall()

    cur.close()
    conn.close()
    return results

def debug_db_stats():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM documents;")
    doc_count = cur.fetchone()[0]

    cur.execute("SELECT AVG(vector_dims(embedding)) FROM documents;")
    dims = cur.fetchone()[0]

    cur.close()
    conn.close()

    return doc_count, dims

