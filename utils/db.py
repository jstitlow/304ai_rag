import psycopg2
import psycopg2.extras
import os

DB_NAME = "rag_db"
DB_USER = "rag_user"
DB_PASS = "ragpassword"
DB_HOST = "localhost"
DB_PORT = 5432

# Connect to DB
conn = psycopg2.connect(
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASS,
    host=DB_HOST,
    port=DB_PORT
)
conn.autocommit = True
cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

# Create table if not exists
cur.execute("""
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding VECTOR(768)  -- must match embedding dimension
)
""")

def add_document(content, embedding):
    """
    Insert a document and its embedding into PostgreSQL
    """
    cur.execute(
        "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
        (content, embedding)
    )

def retrieve_similar(query_embedding, top_k=5):
    """
    Retrieve top-k similar documents using cosine distance
    """
    cur.execute("""
    SELECT content, embedding <#> %s AS distance
    FROM documents
    ORDER BY distance
    LIMIT %s
    """, (query_embedding, top_k))
    return cur.fetchall()
