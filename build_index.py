import os
import faiss
import json
import numpy as np
from pathlib import Path
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def normalize(vectors: np.ndarray) -> np.ndarray:
    """L2 normalize vectors for cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-10, None)

def build_faiss_index(processed_dir="processed", index_file="index.faiss", meta_file="metadata.json"):
    texts, metas, vectors = [], [], []

    for jsonl_path in Path(processed_dir).glob("*.jsonl"):
        with open(jsonl_path, "r") as f:
            for line in f:
                obj = json.loads(line)
                text = obj["text"]
                meta = obj["metadata"]

                # 🔑 Create embedding
                emb = client.embeddings.create(
                    model="text-embedding-3-small",  # efficient embedding model
                    input=text
                ).data[0].embedding

                texts.append(text)
                metas.append(meta)
                vectors.append(emb)

    # Convert & normalize vectors
    vectors = np.array(vectors, dtype="float32")
    vectors = normalize(vectors)

    # Build FAISS index using cosine similarity (inner product)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    # Save index + metadata
    faiss.write_index(index, index_file)
    with open(meta_file, "w") as f:
        json.dump({"texts": texts, "metas": metas}, f)

    print(f"✅ Index built with {len(texts)} entries using cosine similarity.")

if __name__ == "__main__":
    build_faiss_index()
