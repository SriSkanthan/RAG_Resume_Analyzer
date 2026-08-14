from pypdf import PdfReader
from docx import Document


def read_resume(upload_file):
    """Read text content from a PDF, DOCX, or TXT resume."""
    file_name = upload_file.name.lower()

    if file_name.endswith(".pdf"):
        pdf_reader = PdfReader(upload_file)
        resume_text = ""

        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                resume_text += page_text + "\n"

        return resume_text

    if file_name.endswith(".docx"):
        document = Document(upload_file)
        resume_text = ""

        for paragraph in document.paragraphs:
            if paragraph.text.strip():
                resume_text += paragraph.text + "\n"
        return resume_text

    if file_name.endswith(".txt"):
        return upload_file.read().decode("utf-8")

    return ""
