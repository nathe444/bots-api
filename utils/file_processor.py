import os
import shutil
import uuid
import logging

logger = logging.getLogger(__name__)

def save_uploaded_file(file, upload_dir):
    """Save an uploaded file to disk and return the file path."""
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(upload_dir, unique_filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return file_path, unique_filename

def extract_content(file_path, file_type):
    """Extract text content from a file based on its type."""
    content = ""
    
    # Truncate file_type to 50 characters to fit in the database column
    if file_type and len(file_type) > 50:
        file_type = file_type[:50]
    
    if file_type == "application/pdf":
        try:
            import PyPDF2
            with open(file_path, "rb") as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                for page_num in range(len(pdf_reader.pages)):
                    content += pdf_reader.pages[page_num].extract_text() + "\n"
        except ImportError:
            logger.warning("PyPDF2 not installed. Cannot extract PDF content.")
            content = "PDF content extraction requires PyPDF2 library."
    elif file_type.startswith("application/vnd.openxmlformats-officedocument"):
        try:
            import docx
            doc = docx.Document(file_path)
            content = "\n".join([para.text for para in doc.paragraphs])
        except ImportError:
            logger.warning("python-docx not installed. Cannot extract DOCX content.")
            content = "DOCX content extraction requires python-docx library."
    elif file_type.startswith("text/"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as text_file:
            content = text_file.read()
    else:
        content = f"Content extraction not supported for {file_type}"
    
    return content, file_type