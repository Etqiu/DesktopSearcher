"""
Query example: Search the Brain index by semantic similarity.
"""
from pathlib import Path
import duckdb
from sentence_transformers import SentenceTransformer
import numpy as np

DB_PATH = Path(__file__).parent.parent / "brain.duckdb"


def search_files(query: str, top_k: int = 5):
    """Search indexed files by semantic similarity."""
    # Load model and generate query embedding
    print(f"🔍 Searching for: '{query}'")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode(query).tolist()
    
    # Connect to DB
    conn = duckdb.connect(str(DB_PATH))
    
    # Get all files with embeddings
    results = conn.execute("""
        SELECT id, filename, path, text_snippet, embedding
        FROM files_index
        WHERE embedding IS NOT NULL
    """).fetchall()
    
    if not results:
        print("No indexed files found.")
        return
    
    # Compute cosine similarity manually
    similarities = []
    for row in results:
        file_id, filename, path, snippet, embedding = row
        if embedding:
            # Cosine similarity
            dot_product = np.dot(query_embedding, embedding)
            norm_query = np.linalg.norm(query_embedding)
            norm_doc = np.linalg.norm(embedding)
            similarity = dot_product / (norm_query * norm_doc)
            similarities.append((similarity, filename, path, snippet))
    
    # Sort by similarity
    similarities.sort(reverse=True, key=lambda x: x[0])
    
    # Display top results
    print(f"\n📊 Top {top_k} results:\n")
    for i, (score, filename, path, snippet) in enumerate(similarities[:top_k], 1):
        print(f"{i}. {filename} (similarity: {score:.3f})")
        print(f"   Path: {path}")
        print(f"   Snippet: {snippet[:150]}...")
        print()
    
    conn.close()


if __name__ == "__main__":
    # Example queries
    search_files("resumes", top_k=3)
    # search_files("python code examples", top_k=3)
    # search_files("data analysis", top_k=5)
