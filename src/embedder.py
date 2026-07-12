import os
os.environ["HF_HOME"] = os.getenv("HF_HOME", "/tmp/hf_cache")

from sentence_transformers import SentenceTransformer
import faiss, numpy as np, pickle

MODEL_NAME = "all-MiniLM-L6-v2"   # small, only 90MB
INDEX_PATH  = "data/index.faiss"
CHUNKS_PATH = "data/chunks.pkl"

print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)
print("Model ready!")

def embed_and_save(chunks: list[dict]):
    """Embed all chunks and save index to disk."""
    if not chunks:
        raise ValueError("No chunks to embed — no supported code files found in this repo.")
    os.makedirs("data", exist_ok=True)
    
    texts = [f"{c['metadata']}\n\n{c['text']}" for c in chunks]
    
    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype('float32'))
    
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, 'wb') as f:
        pickle.dump(chunks, f)
    
    print(f"Index saved! {len(chunks)} chunks indexed.")

def load_index():
    """Load existing index from disk."""
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, 'rb') as f:
        chunks = pickle.load(f)
    return index, chunks, model