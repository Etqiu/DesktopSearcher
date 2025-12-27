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
        
        # Connect to database (default mode to share config with indexer)
        conn = duckdb.connect(str(db_path))
        
        # Optimization: Perform cosine similarity in DuckDB SQL
        # list_cosine_similarity is available in DuckDB 0.10.0+
        # We cast the query embedding to a FLOAT[] literal for the query
        
        # Format the embedding list as a SQL array string: [0.1, 0.2, ...]
        embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"
        
        try:
            results = conn.execute(f"""
                SELECT 
                    list_cosine_similarity(embedding, {embedding_str}::FLOAT[384]) as score,
                    filename, 
                    path, 
                    text_snippet
                FROM files_index
                WHERE embedding IS NOT NULL
                ORDER BY score DESC
                LIMIT 10
            """).fetchall()
            
            # Results are already sorted and limited
            # Format: (score, filename, path, snippet)
            return results
            
        except duckdb.BinderException:
            # Fallback for older DuckDB versions or if list_cosine_similarity is missing
            # Fetch all and compute in Python (slower, higher memory)
            print("Fallback: list_cosine_similarity not found, using Python-side calculation")
            rows = conn.execute("""
                SELECT filename, path, embedding, text_snippet
                FROM files_index
                WHERE embedding IS NOT NULL
            """).fetchall()
            
            similarities = []
            for filename, path, embedding, snippet in rows:
                if embedding:
                    dot_product = np.dot(query_embedding, embedding)
                    norm_query = np.linalg.norm(query_embedding)
                    norm_doc = np.linalg.norm(embedding)
                    
                    if norm_query > 0 and norm_doc > 0:
                        similarity = dot_product / (norm_query * norm_doc)
                        similarities.append((similarity, filename, path, snippet))
            
            similarities.sort(reverse=True, key=lambda x: x[0])
            return similarities[:10]
            
        finally:
            conn.close()
        
    except Exception as e:
        print(f"Search error in module: {e}")
        return []

def get_recent_files(db_path, limit=50):
    """Fetch recently indexed files."""
    try:
        conn = duckdb.connect(str(db_path))
        # Return format matching search results: (score, filename, path, snippet)
        # Score is 1.0 for recent files
        results = conn.execute(f"""
            SELECT 
                1.0 as score,
                filename, 
                path, 
                text_snippet
            FROM files_index
            ORDER BY created_at DESC
            LIMIT {limit}
        """).fetchall()
        conn.close()
        return results
    except Exception as e:
        print(f"Error fetching recent files: {e}")
        return []
