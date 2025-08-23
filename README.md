---
title: Assistant Teacher
emoji: 📚
colorFrom: gray
colorTo: green
sdk: gradio
sdk_version: 5.43.1
app_file: app.py
pinned: false
license: mit
short_description: An assistant for secondary school teacher
---

# CBSE RAG Tutor

A Retrieval-Augmented Generation (RAG) system designed to assist teachers and students with CBSE curriculum content by extracting text from educational materials, indexing it, and providing intelligent responses to questions about the curriculum.

## Overview

This project processes educational materials (PDFs) into a searchable knowledge base and provides an AI-powered interface for querying this information. The system consists of three main components:

1. **Data Ingestion Pipeline** - Extracts and processes text from PDF textbooks
2. **Vector Index Creation** - Builds a searchable FAISS index from processed content
3. **Query Interface** - Allows users to search for information and generate responses

## Features

- Extract text from PDF textbooks using OCR
- Process text into searchable chunks with metadata
- Build vector embeddings for semantic search
- Answer curriculum-based questions
- Generate practice questions on specific topics
- Provide explanations of concepts

## Prerequisites

- Python 3.8+
- OpenAI API key
- OCR dependencies:
  - Poppler (for PDF processing)
  - Tesseract (for text extraction)
  - EasyOCR
- SpaCy with English language model
- FAISS for vector search

## Expected directory structure
```
teacher-assist/
├── input/                  # Place your PDF textbooks here in a structured format
│   ├── grade_4/            # Format: grade_X where X is the grade level
│   │   ├── science/        # Subject name (lowercase)
│   │   │   ├── plants/     # Chapter name (lowercase)
│   │   │   │   ├── plants.pdf  # PDF files
│   │   │   ├── animals/
│   │   │   │   ├── animals.pdf
│   │   ├── mathematics/
│   │   │   ├── fractions/
│   │   │   │   ├── fractions.pdf
│   ├── grade_5/
│   │   ├── ...
│
├── work-area/              # Temporary storage for extracted text files
│   ├── grade_4/
│   │   ├── science/
│   │   │   ├── plants/
│   │   │   │   ├── plants.txt  # OCR output from PDFs
│
├── processed/              # Chunked and processed JSONL files 
│   ├── 4_science_plants_plants.jsonl  # Format: {grade}_{subject}_{chapter}_{filename}.jsonl
│   ├── ...
```
## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/cbse-rag-tutor.git
cd cbse-rag-tutor

# Install required Python packages
pip install -r requirements.txt

# Install spaCy English model
python -m spacy download en_core_web_sm

# Install Poppler (macOS)
brew install poppler

# Install Tesseract (macOS)
brew install tesseract

```
## Usage
### Data Ingestion

1. Place your PDF textbooks in the `data/` directory.
2. Run the ingestion script to extract text and metadata:
   ```bash
   python ingest_v2.py
   ```

### Index Creation

1. Build the vector index from the ingested data:
   ```bash
   python build_index.py
   ```

### Query Interface

1. Start the query interface:
   ```bash
   python teacher_assist.py
   ```

2. Use the interface to ask questions about the curriculum or generate practice questions.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for discussion.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.