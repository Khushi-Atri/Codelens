from rank_bm25 import BM25Okapi
import numpy as np
from src.embedder import load_index

def reciprocal_rank_fusion(vector_results, bm25_results, k=60):
    scores = {}
    for rank, (chunk, _) in enumerate(vector_results):
        key = f"{chunk['filepath']}:{chunk['start_line']}"
        scores[key] = scores.get(key, 0) + 1 / (k + rank)
    for rank, (chunk, _) in enumerate(bm25_results):
        key = f"{chunk['filepath']}:{chunk['start_line']}"
        scores[key] = scores.get(key, 0) + 1 / (k + rank)
    return scores

def search(query: str, top_k: int = 6) -> list[dict]:
    index, chunks, model = load_index()
    
    # Vector search
    q_vec = model.encode([query], normalize_embeddings=True)
    distances, indices = index.search(
        np.array(q_vec, dtype='float32'), top_k * 3
    )
    vector_results = [
        (chunks[i], float(distances[0][j]))
        for j, i in enumerate(indices[0])
        if i < len(chunks)
    ]
    
    # BM25 keyword search
    tokenized_corpus = [c['text'].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(query.lower().split())
    top_bm25_idx = np.argsort(bm25_scores)[::-1][:top_k * 3]
    bm25_results = [
        (chunks[i], float(bm25_scores[i]))
        for i in top_bm25_idx
    ]
    
    # Combine with RRF
    rrf_scores = reciprocal_rank_fusion(vector_results, bm25_results)
    all_chunks_map = {}
    for chunk, _ in vector_results + bm25_results:
        key = f"{chunk['filepath']}:{chunk['start_line']}"
        all_chunks_map[key] = chunk
    
    sorted_keys = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
    return [all_chunks_map[k] for k in sorted_keys[:top_k]]