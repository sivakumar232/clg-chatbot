import re
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def is_mqp_pdf(path: Path) -> bool:
    """Return True if the PDF is a Model Question Paper or model paper."""
    name = path.name.lower().replace(" ", "_").replace("-", "_")
    mqp_keywords = ["mqp", "model_paper", "model_question", "modelquestion", "model_papers"]
    return any(kw in name for kw in mqp_keywords)


def is_cover_page(text: str) -> bool:
    """Detect if a PDF page is a repetitive cover page containing college headers but no course content."""
    text_upper = text.upper()
    has_header = (
        "SAGI RAMA KRISHNAM RAJU" in text_upper
        or "BHIMAVARAM" in text_upper
        or "ESTD:1980" in text_upper
    )
    has_course_code = bool(re.search(r"B\d{2}[A-Z]{2,4}\d{4}", text))
    has_unit = (
        "UNIT-" in text_upper
        or "UNIT I" in text_upper
        or "COURSE OBJECTIVES" in text_upper
        or "COURSE OUTCOMES" in text_upper
    )
    has_structure = (
        "STRUCTURE" in text_upper
        or "SCHEME OF" in text_upper
        or "SEMESTER" in text_upper
    )
    return has_header and not (has_course_code or has_unit or has_structure)


def load_pdf_files(data_dir: str) -> list[Document]:
    """
    Scans the given directory and its subdirectories for all .pdf files,
    skipping MQP / Model Question Papers and cover pages.
    Returns a list of LangChain Document objects.
    """
    print(f"Scanning for PDF files in {data_dir}...")
    all_pdfs = sorted(Path(data_dir).rglob("*.pdf"))

    valid_pdfs = []
    skipped_count = 0
    for pdf in all_pdfs:
        if is_mqp_pdf(pdf):
            skipped_count += 1
        else:
            valid_pdfs.append(pdf)

    print(
        f"Found {len(all_pdfs)} total PDFs. Skipping {skipped_count} MQP/model paper PDFs."
    )

    docs = []
    skipped_cover_pages = 0
    for pdf in valid_pdfs:
        try:
            loader = PyPDFLoader(str(pdf))
            file_pages = loader.load()
            for page_idx, page_doc in enumerate(file_pages):
                # Check first 2 pages for repetitive cover page
                if page_idx < 2 and is_cover_page(page_doc.page_content):
                    skipped_cover_pages += 1
                    continue
                docs.append(page_doc)
        except Exception as e:
            print(f"  ⚠️ Warning: Failed to load {pdf}: {e}")

    print(
        f"Successfully loaded {len(docs)} PDF pages from {len(valid_pdfs)} PDFs "
        f"(skipped {skipped_cover_pages} cover pages)."
    )
    return docs
