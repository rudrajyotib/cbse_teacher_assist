import numpy as np
import faiss
import json
import os
from pathlib import Path
from openai import OpenAI
import time

# Initialize OpenAI client
client = OpenAI()

# Define embedding model
EMBED_MODEL = "text-embedding-3-small"

# Load the index and data
def load_data(index_path="index.faiss", metadata_path="metadata.json"):
    """Load the FAISS index and metadata"""
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index file not found: {index_path}")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    # Load FAISS index
    index = faiss.read_index(index_path)

    # Load metadata which contains both texts and metas
    with open(metadata_path, 'r') as f:
        data = json.load(f)

    # Extract the texts and metadata arrays
    texts = data.get("texts", [])
    metas = data.get("metas", [])

    if not texts or not metas:
        raise ValueError("Missing texts or metadata in the metadata file")

    return index, texts, metas

# Load the necessary data
try:
    index, texts, metas = load_data()
    print(f"Loaded index with {index.ntotal} vectors")
    print(f"Loaded {len(texts)} text chunks and {len(metas)} metadata entries")
except Exception as e:
    print(f"Error loading data: {e}")

def search(query, k=3, filters=None):
    """Search for relevant text chunks based on the query (cosine similarity)."""
    try:
        # 🔑 Embed query
        response = client.embeddings.create(
            model=EMBED_MODEL,
            input=query
        )
        q_embed = np.array([response.data[0].embedding], dtype="float32")
        # normalize query
        q_embed /= np.linalg.norm(q_embed, axis=1, keepdims=True)
        print("✅ Query embedding created & normalized")
    except Exception as e:
        print(f"❌ Error creating embedding: {e}")
        return []

    # 🔍 Search in FAISS
    similarities, indices = index.search(q_embed, k * 3)  # oversample to apply filters
    top_score = similarities[0][0]
    print(f"🔎 Top similarity score: {top_score:.2f}")

    # Adjust threshold (cosine sim: 0.0–1.0)
    if top_score < 0.55:
        print(f"⚠️ No relevant chunks found (top similarity {top_score:.2f} < 0.55)")
        return []

    results = []
    for i, idx in enumerate(indices[0]):
        if idx < 0 or idx >= len(texts):  # Guard against out of bounds
            continue

        meta = metas[idx]
        text = texts[idx]

        # Apply filters if given
        if filters:
            skip = False
            for key, val in filters.items():
                if meta.get(key) != val:
                    skip = True
                    break
            if skip:
                continue

        results.append((meta, text))
        if len(results) >= k:
            break

    return results

def answer_question(query, filters=None):
    """Generate an answer based on relevant text chunks"""
    hits = search(query, k=6, filters=filters)

    if not hits:
        return "I couldn't find any relevant information to answer your question."

    context = "\n\n".join([f"{h[1]}" for h in hits])

    # Include metadata information for debugging
    context_info = "\n".join([
        f"- {h[0]['chunk_id']} (grade: {h[0]['grade']}, subject: {h[0]['subject']}, chapter: {h[0]['chapter']})"
        for h in hits
    ])
    print(f"Using these chunks:\n{context_info}")

    prompt = f"""
    You are a helpful teacher assistant.
    Use the following textbook excerpts to answer the question.

    Context:
    {context}

    Question:
    {query}

    Answer in a clear and simple way suitable for a grade {hits[0][0]['grade']} student.
    """

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "Sorry, I encountered an error while generating your answer."

def explain_concept(query, filters=None):
    """Generate an explanation for a concept based on relevant text chunks"""
    hits = search(query, k=6, filters=filters)

    if not hits:
        return "I couldn't find any relevant information to explain this concept."

    context = "\n\n".join([f"{h[1]}" for h in hits])

    # Include metadata information for debugging
    context_info = "\n".join([
        f"- {h[0]['chunk_id']} (grade: {h[0]['grade']}, subject: {h[0]['subject']}, chapter: {h[0]['chapter']})"
        for h in hits
    ])
    print(f"Using these chunks:\n{context_info}")

    prompt = f"""
    You are a helpful teacher assistant.
    Use the following textbook excerpts to explain the concept.

    Context:
    {context}

    Explain the concept of "{query}" in a clear and simple way suitable for a grade {hits[0][0]['grade']} student.
    """

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"Error generating explanation: {e}")
        return "Sorry, I encountered an error while generating your explanation."

def create_question(query, filters=None, questionCount = 5):
    """Generate an answer based on relevant text chunks"""
    hits = search(query, k=6, filters=filters)

    if not hits:
        return "I couldn't find any relevant information to answer your question."

    context = "\n\n".join([f"{h[1]}" for h in hits])

    # Include metadata information for debugging
    context_info = "\n".join([
        f"- {h[0]['chunk_id']} (grade: {h[0]['grade']}, subject: {h[0]['subject']}, chapter: {h[0]['chapter']})"
        for h in hits
    ])
    print(f"Using these chunks:\n{context_info}")

    prompt = f"""
    You are a helpful teacher assistant.
    Use the following textbook excerpts to answer the question.
    Your task is to create {questionCount} simple and clear questions based on the provided context on the topic of "{query}".
    Questions should be suitable for a grade {hits[0][0]['grade']} student.
    You should also provide the answers to these questions.
    The format should be as follows:
    Question 1: <Your first question>
    Answer 1: <Answer to the first question>
    Question 2: <Your second question>
    Answer 2: <Answer to the second question>
    
    You must adhere to the context provided below.

    Context:
    {context}

    
    """

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "Sorry, I encountered an error while generating your answer."

if __name__ == "__main__":
    # Test the search functionality
    print("Testing search functionality...")
    query = "food chain"
    filters = {"grade": "4", "subject": "science"}

    hits = search(query, k=2, filters=filters)
    print(f"Found {len(hits)} relevant chunks")

    for i, (meta, text) in enumerate(hits):
        print(f"\nHit {i+1}:")
        print(f"Metadata: {meta['grade']}, {meta['subject']}, {meta['chapter']}")
        print(f"Text snippet: {text[:200]}...")

    # Test the answer generation
    print("\nGenerating explanation to a sample topic...")
    answer = explain_concept(query, filters=filters)
    print("\nExplanation:")
    print(answer)
    print("\nGenerated Questions:")
    questions = create_question(query, questionCount=3, filters=filters)
    print(questions)