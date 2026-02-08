from pypdf import PdfReader
import os

def load_document(source):
    """
    Accepts either:
    - Streamlit UploadedFile
    - file path string
    """

    # -------- Uploaded file --------
    if hasattr(source, "type"):
        if source.type == "text/plain":
            return source.read().decode("utf-8")

        elif source.type == "application/pdf":
            reader = PdfReader(source)
            return "\n".join(page.extract_text() or "" for page in reader.pages)

    # -------- File path --------
    elif isinstance(source, str):
        ext = os.path.splitext(source)[1].lower()

        if ext == ".txt":
            with open(source, "r", encoding="utf-8") as f:
                return f.read()

        elif ext == ".pdf":
            reader = PdfReader(source)
            return "\n".join(page.extract_text() or "" for page in reader.pages)

    raise ValueError(f"Unsupported document source: {source}")
