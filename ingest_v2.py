import os, re, json
from pathlib import Path
from PIL import Image

import fitz
from pdf2image import convert_from_path
import pytesseract
from transformers import GPT2TokenizerFast
import spacy
import faiss, json, os, numpy as np
# Tokenizer + NLP
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
nlp = spacy.load("en_core_web_sm")
import easyocr



# --- Cleaning ---
def clean_text(text: str) -> str:
    text = re.sub(r'Page\s*\d+', '', text)        # page numbers
    text = re.sub(r'Figure\s*\d+.*', '', text)    # figure labels
    text = re.sub(r'(Test Yourself|Exercise|SELF TEST).*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(MtG Olympiad).*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(Class-\d+ |)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(Olympiad Bite)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\n+', '\n', text)             # normalize newlines
    return text.strip()

def extract_text_or_image_as_text_from_pdf(path, reader, out_txt, dpi=200):
    """
    Extract text from a PDF by using direct extraction for text pages
    and OCR for image-based pages.

    Args:
        path: Path to the PDF file
        reader: EasyOCR reader instance
        dpi: DPI for image conversion

    Returns:
        List of extracted text strings, one per page
    """
    doc = fitz.open(path)
    texts = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text().strip()
        print(f"Processing page {page_num + 1}/{len(doc)}: {len(text)} chars of text")

        # If the page has text content
        if text:
            texts.append(text)
        else:
            print(f"Page {page_num + 1} has no text, using OCR")
            # Render the page as an image and use OCR
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Apply OCR to the image
            res = reader.readtext(np.array(img), detail=0, paragraph=True)
            page_text = "\n".join([r.strip() for r in res if r and r.strip()])
            print(f"OCR extracted {len(page_text)} chars from page {page_num + 1}")
            print(page_text)
            texts.append(page_text)

    doc.close()

    cleaned = clean_text("\n".join(texts))
    Path(out_txt).parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(cleaned)
    return out_txt


# --- OCR ---
def pdf_to_text(pdf_path: str, out_txt: str):
    pages = convert_from_path(pdf_path, dpi=300)
    text = []
    for page in pages:
        text.append(pytesseract.image_to_string(page, lang="eng"))
    cleaned = clean_text("\n".join(text))
    Path(out_txt).parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(cleaned)
    return out_txt

# --- Chunking ---
def chunk_paragraphs(text, max_tokens=300, overlap=50):
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 20]
    chunks = []
    for para in paragraphs:
        tokens = tokenizer.encode(para)
        if len(tokens) <= max_tokens:
            chunks.append(para)
        else:
            sentences = re.split(r'(?<=[.!?]) +', para)
            current, token_count = [], 0
            for sent in sentences:
                sent_tokens = tokenizer.encode(sent)
                if token_count + len(sent_tokens) > max_tokens:
                    chunks.append(" ".join(current))
                    # Start new chunk with overlap
                    overlap_text = " ".join(current)[-overlap:]
                    current = [overlap_text] if overlap_text else []
                    token_count = len(tokenizer.encode(overlap_text))
                current.append(sent)
                token_count += len(sent_tokens)
            if current:
                chunks.append(" ".join(current))
    return chunks

# --- Topic Extraction ---
def extract_topics(paragraph: str, top_k=5):
    doc = nlp(paragraph)
    candidates = [chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text) > 2]
    return list(set(candidates))[:top_k]

# --- Main Process (from work-area txt -> JSONL) ---
def process_text_file(txt_path: Path, source_file_name: str, output_dir="processed"):
    parts = txt_path.parts
    grade, subject, chapter, file_name = parts[-4].split('_')[1], parts[-3], parts[-2], parts[-1]
    file_name = file_name.replace(".txt", "")

    with open(txt_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    chunks = chunk_paragraphs(raw_text)

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{grade}_{subject}_{chapter}_{file_name}.jsonl")

    with open(output_file, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            topics = extract_topics(chunk)
            record = {
                "text": chunk,
                "metadata": {
                    "grade": grade,
                    "subject": subject,
                    "chapter": chapter,
                    "topics": topics,
                    "source_file": source_file_name,
                    "chunk_id": f"{grade}_{subject}_{chapter}_{file_name}_c{i+1}"
                }
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output_file

def run_recursive_pipeline(root_dir, work_area="work-area", processed="processed"):
    """
    Walk root_dir recursively, find PDFs, OCR them to work-area,
    then process each txt into processed JSONL.
    """
    root_dir = Path(root_dir)
    reader = easyocr.Reader(['en'], gpu=False)
    for pdf_path in root_dir.rglob("*.pdf"):
        try:
            print(f"📖 Processing {pdf_path} ...")
            rel_parts = pdf_path.parts[-4:]  # grade/subject/chapter/chapter.pdf
            txt_path = Path(work_area, *rel_parts).with_suffix(".txt")

            # Step 1: OCR -> TXT
            if not txt_path.exists():  # don’t redo OCR if already done
                print(f"  🔍 OCR -> {txt_path}")
                extract_text_or_image_as_text_from_pdf(pdf_path, reader, txt_path)
            else:
                print(f"  ✅ Skipping OCR, already exists: {txt_path}")

            # Step 2: TXT -> JSONL
            print(f"  ✂️  Chunking -> JSONL")
            process_text_file(txt_path, pdf_path.name,  processed)
            print(f"  ✅ Done: {pdf_path.name}")

        except Exception as e:
            print(f"❌ Error processing {pdf_path}: {e}")

if __name__ == "__main__":
    # Assume your textbooks live under ./data/
    run_recursive_pipeline("input", work_area="work-area", processed="processed")