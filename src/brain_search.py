import duckdb
import numpy as np
from pathlib import Path

def perform_search(query, model, db_path):
    """
    Executes semantic search against the DuckDB index.
    
    Args:
        query (str): The search query.
        model (SentenceTransformer): The loaded embedding model.
        db_path (Path): Path to the DuckDB database.
        
    Returns:
        list: Top 10 results as tuples (similarity, filename, path).
    """
    try:
        query_embedding = model.encode(query).tolist()
        
        # Use read_only=True to avoid locking conflicts
        conn = duckdb.connect(str(db_path), read_only=True)
        results = conn.execute("""
            SELECT filename, path, embedding
            FROM files_index
            WHERE embedding IS NOT NULL
        """).fetchall()
        conn.close()
        
        similarities = []
        for filename, path, embedding in results:
            if embedding:
                dot_product = np.dot(query_embedding, embedding)
                norm_query = np.linalg.norm(query_embedding)
                norm_doc = np.linalg.norm(embedding)
                
                if norm_query > 0 and norm_doc > 0:
                    similarity = dot_product / (norm_query * norm_doc)
                    similarities.append((similarity, filename, path))
        
        similarities.sort(reverse=True, key=lambda x: x[0])
        return similarities[:10]
        
    except Exception as e:
        print(f"Search error in module: {e}")
        return []
