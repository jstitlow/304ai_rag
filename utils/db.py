import psycopg2
import psycopg2.extras
import os
from utils.loader import load_document
from utils.embedding import chunk_text, get_embedding
from init_db import get_conn

