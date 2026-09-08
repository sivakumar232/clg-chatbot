# clg-chatbot
Chunking Analysis — SRKR RAG Pipeline
Based on direct observation of raw MD files, raw PDF pages, and actual chunk output.

What the Data Actually Looks Like
Markdown Files (189 files)
These are scraped website pages. When you look at the raw content, they have severe structural problems:

Problem 1 — Boilerplate noise is massive

Every single .md file starts and ends with identical garbage:


Source URL: https://www.srkrec.ac.in/...
--------------------------------------------------
Student Clubs | SRKR Engineering College
* [Home](/)
* [Life @ SRKR](#)
...and ends with:


SRKR Classic  Executive Navy  Heritage Maroon  Emerald Slate  Royal Violet...
⚙ Theme Presets  Reset  Close
0 online now
Copyright © 2026 All Rights Reserved by SRKR Engineering College
![SRKR Engineering College](/assets/images/logo/srkr-logo.png)
Apply to SRKR – Start your journey with excellence.
Mobile: +91 9848823332  info@srkrec.ac.in
Follow Us:
This boilerplate is the entire nav bar + footer + theme switcher repeated across all 189 files. The current chunker does not strip any of this. So chunks like this are being embedded:


[148 chars] 'SRKR Classic Executive Navy Heritage Maroon Emerald Slate Royal Violet...'
[99 chars]  'Programs Offered | B.Tech, M.Tech, MBA, MCA | SRKR Engineering College\n* [Home](/)\n* [Academics](#)'
[187 chars] '![TCS](/assets/images/recruiters/tcs.png)\n![Infosys](</assets/images/...'
These are pure noise. A user asking "what is the CSE fee?" could retrieve a nav-bar chunk instead of the actual fee information.

Problem 2 — Image tags become meaningless text

placements-summary.md is full of:


![SHANMUKHI GUDAPATI](/assets/images/placements/students/SHANMUKHIGUDAPATI.jpg>)
![Infosys](/assets/images/recruiters/infosys.png)
19 LPA
SHANMUKHI GUDAPATI
Each student's placement is a 3-line fragment. Chunked at 1000 chars, these get merged together randomly into one block of 15 student names + logos with no semantic meaning as a unit.

Problem 3 — Header-split chunks lose context

The MarkdownHeaderTextSplitter splits on #, ##, ###. So:

markdown

## 26
Members across 6 categories
becomes chunk: "Members across 6 categories" with metadata Header 2: "26".

A user asking "how many members does the academic council have?" will NOT find this — because the number 26 is in the metadata header, not the text.

PDF Files (515 files)
The PDF content has completely different problems:

Problem 1 — Cover pages become useless chunks

Every syllabus PDF starts with:


SAGI RAMA KRISHNAM RAJU ENGINEERING COLLEGE (AUTONOMOUS)
(Affiliated to JNTUK, Kakinada), (Recognized by AICTE, New Delhi)
Accredited by NAAC with 'A' Grade
UG Programmes CE,CSE,ECE,EEE,IT & ME are Accredited by NBA
Chinna Amiram, Bhimavaram-534204. (AP)
Estd:1980
1,111 characters of pure header. This is embedded as a chunk. Every single PDF has this. You are embedding the same college header text hundreds of times.

Problem 2 — Page numbers bleed into text

Raw PDF pages start with "1 \n \n \nSAGI RAMA..." and "2 \n \nCourse Code...". The page number 1, 2, 3 etc. are extracted as part of the text because PyPDFLoader extracts text linearly. After chunking:

RecursiveCharacterTextSplitter at 1000 chars just cuts across these page boundaries blindly
A single 1000-char chunk can span the bottom of page 3 and top of page 4, mixing two unrelated subjects
Problem 3 — Tabular syllabus data becomes unreadable

Syllabus PDF pages contain dense tables:


Course Code Category  L  T  P  C  I.M  E.M  Exam
B20HS1101    HS       3  -- --  3   30   70  3Hrs
ENGLISH
(Common to AIDS,CE,CSE,ECE,EEE,IT&ME)
Introduction:
The course is designed to train students...
After RecursiveCharacterTextSplitter at 1000 chars: the table header, the course intro text, and the next course's table rows all get merged into one chunk. A user asking "what is the credit for English B20HS1101?" will not get a clean answer because 3 credits is in a table row, and the chunk around it is 1000 chars of mixed content.

Problem 4 — Model Question Papers (MQPs) give zero RAG value

Model papers chunks look like:


Page 2 of 32
human order.
  OR    
10 a). Case study of typical holistic technologies. 5 3 7 
 b). Role of engineer in promoting harmony in society 5 3 7 
CO-COURSE OUTCOME  KL-KNOWLEDGE LEVEL  M-MARKS
NOTE: Questions can be given as A,B splits or as a single Question for 14 marks
This is an exam question stub. When a student asks "Tell me about R20 curriculum", they should NOT get a list of exam question marks tables. Yet MQPs make up ~50% of the R20 folder.

Why the Current Strategy Fails
Issue	Current Code	Impact
Nav/footer boilerplate in MD	Not stripped	~30% of MD chunks are pure nav noise
Image alt-text in MD	Kept as text	![Infosys](path) embedded as meaningful content
Header-text split loses numbers	MarkdownHeaderTextSplitter	Key facts (stats, counts) land in metadata, not chunk text
PDF cover pages repeated	No deduplication	Same college header embedded hundreds of times
PDF page numbers in text	PyPDFLoader line-by-line	"2\n\nCourse Code..." fragments
PDF cross-page cuts	Fixed 1000-char window	Subject content arbitrarily split mid-topic
MQP exam questions in syllabus PDFs	No file-type routing	Exam paper noise pollutes curriculum retrieval
Recommended Strategy per Data Type
For Markdown (Website Pages) — Pre-clean before chunking
Step 1: Strip boilerplate before chunking

Add a pre-processing step that removes:

Everything before the first # heading (nav bar)
Everything after the Copyright © or Follow Us: line (footer)
All ![...]() image tags (alt text + broken paths are meaningless)
All * [Home](/), * [Life @ SRKR](#) navigation breadcrumbs
Theme switcher lines
This alone would cut MD chunk count by ~30% and eliminate all navigation noise.

Step 2: Prepend source context into each chunk

The source_url is in metadata but not in the chunk text. When the vector search retrieves a chunk, the LLM needs context about where it came from. Prepend it:

python

chunk_text = f"[Source: SRKR {page_title}]\n\n{chunk_text}"
Step 3: Keep using MarkdownHeaderTextSplitter but fix number-only headers

When a header is purely numeric (## 26, ## 1339+), merge it into the next text block so the number lands in the chunk text, not just metadata.

For PDFs — Completely different approach needed per PDF type
There are 3 distinct PDF types in your data that need 3 different strategies:

Type A — Syllabus PDFs (R20, R23, R24 curriculum documents)

These are dense, structured academic documents. RecursiveCharacterTextSplitter at 1000 chars is too blunt.

Better strategy:

Chunk by course unit, not by character count. Each course in the syllabus has a header like ENGLISH, MATHEMATICS-I, PROGRAMMING IN C. Split on these.
Use a larger chunk size (1500-2000 chars) to capture the full unit description in one chunk
Strip cover pages (detect by: page has no course code, just college header → skip)
Skip MQP (model question paper) files entirely — they have zero retrieval value for a college chatbot
Type B — IQAC, BOS, placement reports (structured reports)

These are already text-rich documents with paragraphs.

Better strategy:

RecursiveCharacterTextSplitter at 1200 chars with 200 overlap works reasonably here
The overlap is critical — ensure chunk_overlap=200 so facts that span paragraph boundaries aren't lost
Add the PDF filename as context prefix to each chunk so the LLM knows it's from an IQAC report
Type C — Brochures, newsletters (mixed layout)

These are problematic because PDF layout extraction produces scrambled column order.

Better strategy:

These are lower priority — consider excluding newsletters and keeping only the 1-2 page brochures
If kept, use a smaller chunk size (600-800 chars) since their "paragraphs" extracted by PyPDF are already short
Concrete Changes to Make
1. Add a Markdown cleaner (highest priority)
python

import re
def clean_markdown(text: str) -> str:
    # Remove nav breadcrumb lines
    text = re.sub(r'^\* \[.*?\]\(.*?\)\n', '', text, flags=re.MULTILINE)
    # Remove image tags
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    # Remove footer boilerplate
    cutoff = re.search(
        r'(Copyright ©|Follow Us:|Theme Presets|SRKR Classic|⚙|online now)',
        text
    )
    if cutoff:
        text = text[:cutoff.start()]
    # Remove the scraped page title line (first line like "Student Clubs | SRKR...")
    text = re.sub(r'^.*?\| SRKR Engineering College\n', '', text)
    # Collapse multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()
2. Skip MQP PDFs
python

def should_skip_pdf(path: Path) -> bool:
    name = path.name.lower()
    return any(k in name for k in ['mqp', 'model_paper', 'model-paper', 'model_questions'])
3. Strip PDF cover pages
python

def is_cover_page(text: str) -> bool:
    # Cover pages have college header but no course code
    has_header = 'SAGI RAMA KRISHNAM RAJU' in text
    has_course = bool(re.search(r'B\d{2}[A-Z]{2}\d{4}', text))  # e.g. B20HS1101
    return has_header and not has_course
4. Use larger chunk size for syllabus PDFs
python

# In pdf_strategy.py — route by subdirectory
def chunk_pdfs(docum    ents, chunk_size=1200, chunk_overlap=200):
    ...
Expected Outcome After Fixes
Metric	Current	After Fixes
Total chunks	25,956	~8,000–12,000
Junk chunks (nav/cover/MQP)	~6,000+	~0
Avg chunk quality	Low	High
API calls to Jina	~811	~250–375
Retrieval accuracy	Poor (nav noise drowns answers)	Significantly better
