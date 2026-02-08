from pypdf import PdfReader
from pathlib import Path
import io


def load_document(source):
    """
    Returns raw text from txt or pdf.

    Accepts either:
    - Streamlit UploadedFile
    - file path string / Path
    """

    # =========================================================
    # CASE 1 — Streamlit UploadedFile (your original behavior)
    # =========================================================
    if hasattr(source, "type") and hasattr(source, "read"):
        if source.type == "text/plain":
            return source.read().decode("utf-8")

        elif source.type == "application/pdf":
            reader = PdfReader(source)
            text = ""

            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

            return text

        else:
            raise ValueError(f"Unsupported file type: {source.type}")

    # =========================================================
    # CASE 2 — File path (new persistent data folder support)
    # =========================================================
    path = Path(source)

    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")

    elif path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        text = ""

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        return text

    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")
