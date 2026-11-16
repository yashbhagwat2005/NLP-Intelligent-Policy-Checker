"""Read an uploaded file (.txt, .pdf or .docx) and return its plain text."""
import io
import os

ALLOWED = {".txt", ".pdf", ".docx"}


def read_upload(file_storage):
    """file_storage is the object Flask gives us for an uploaded file."""
    name = os.path.basename(file_storage.filename or "")
    ext = os.path.splitext(name)[1].lower()
    if ext not in ALLOWED:
        raise ValueError(f'"{name}" is not supported. Please upload .txt, .pdf or .docx files.')
    data = file_storage.read()

    if ext == ".txt":
        text = data.decode("utf-8", errors="replace")
    elif ext == ".pdf":
        from pypdf import PdfReader
        pages = PdfReader(io.BytesIO(data)).pages
        text = "\n".join(page.extract_text() or "" for page in pages)
    else:  # .docx
        from docx import Document
        text = "\n".join(p.text for p in Document(io.BytesIO(data)).paragraphs)

    if not text.strip():
        raise ValueError(f'No text could be read from "{name}". Scanned PDFs are not supported.')
    title = os.path.splitext(name)[0].replace("_", " ")
    return title, text
