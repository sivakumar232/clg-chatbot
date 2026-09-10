# Ingestion/processor.py


# Take loaded Document -> clean them -> enrich metadata -> return clean Documents 
import re
from langchain_core.documents import Document


def clean_text(text: str) -> str:
    """Clean text and remove invalid Unicode characters."""

    # Remove invalid UTF-16 surrogate characters
    text = text.encode("utf-8", errors="replace").decode("utf-8")

    text = text.replace("\x00", " ")

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

def clean_markdown_text(text: str) -> tuple[str, str | None]:
    """Clean markdown text by removing site nav, footer, image tags, and theme noise."""
    lines = text.splitlines()
    source_url = None
    if lines and lines[0].startswith("Source URL:"):
        source_url = lines[0].replace("Source URL:", "").strip()
        text = "\n".join(lines[2:])

    # 1. Truncate at footer cutoff point
    footer_pattern = re.compile(
        r"(Copyright\s*©|Follow\s*Us:|Theme\s*Presets|SRKR\s*Classic|⚙\s*Theme|0\s*online\s*now|Apply\s*to\s*SRKR\s*–|Mobile:\s*\+91\s*9848823332|info@srkrec\.ac\.in)",
        re.IGNORECASE,
    )
    m = footer_pattern.search(text)
    if m:
        text = text[: m.start()]

    # 2. Remove nav breadcrumbs and raw scraped site headers
    text = re.sub(r"^\*\s+\[.*?\]\(.*?\)\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^.*?\|\s*SRKR\s*Engineering\s*College\s*$", "", text, flags=re.MULTILINE)

    # 3. Remove image tags ![alt](url)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)

    # 4. Remove theme preset / footer remnant text
    text = re.sub(
        r"(Executive Navy|Heritage Maroon|Emerald Slate|Royal Violet|Reset\s+Close)",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # 5. Convert number-only / stat headers (e.g. ## 26, ## 1339+, ## 19.00 LPA) into bold text
    # so MarkdownHeaderTextSplitter doesn't isolate numbers into metadata while stripping them from text
    num_header_pattern = re.compile(
        r"^(\#+)\s+([\d\.,\+\-%\$\s]|LPA|PM|K|M|Lakhs|Crores)+$",
        re.IGNORECASE | re.MULTILINE,
    )
    text = num_header_pattern.sub(
        lambda m: f"**{m.group(0).lstrip('#').strip()}**",
        text,
    )

    # 6. Remove dangling empty headers
    text = re.sub(r"^\#+\s*$", "", text, flags=re.MULTILINE)

    # 7. Normalize newlines
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    return text, source_url


def process_documents(
    documents: list[Document],
    doc_type: str,
) -> list[Document]:

    processed = []

    for doc in documents:

        if not isinstance(doc.page_content, str):
            print(
                f"Skipping invalid document: "
                f"{type(doc.page_content).__name__}"
            )
            continue

        text = doc.page_content
        metadata = dict(doc.metadata)

        # --------------------------------------------------
        # Markdown / Website
        # --------------------------------------------------
        if doc_type == "markdown":

            text, extracted_url = clean_markdown_text(text)

            if extracted_url:
                metadata["source_url"] = extracted_url
            else:
                metadata["source_url"] = metadata.get(
                    "source_url",
                    "Unknown Webpage",
                )

        # --------------------------------------------------
        # PDF
        # --------------------------------------------------
        elif doc_type == "pdf":

            source = metadata.get("source", "Unknown PDF")
            filename = source.split("/")[-1]

            metadata["source_url"] = f"PDF: {filename}"

        # --------------------------------------------------
        # Metadata
        # --------------------------------------------------

        source_text = (
            metadata.get("source_url", "")
            + " "
            + metadata.get("source", "")
        ).lower()

        department_match = re.search(
            r"/(cse|it|ece|mech|eee|civil|ai&ds)/",
            source_text,
        )

        if department_match:
            metadata["category"] = department_match.group(1)
        else:
            metadata["category"] = "general"

        metadata["content_type"] = doc_type

        # --------------------------------------------------
        # Final cleanup
        # --------------------------------------------------

        text = clean_text(text)

        if not text:
            continue

        processed.append(
            Document(
                page_content=text,
                metadata=metadata,
            )
        )

    print(
        f"Processed {len(processed)} / {len(documents)} {doc_type} documents"
    )

    return processed