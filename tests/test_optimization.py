import pytest
import duckdb
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from brain_search import perform_search
from app import BrainIndexer

# Mock model to avoid loading the real one (slow) or use a tiny one
class MockModel:
    def encode(self, text, show_progress_bar=False):
        # Return a random 384-dim vector
        return np.random.rand(384).astype(np.float32)

@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test_brain.duckdb"

@pytest.fixture
def indexer(db_path):
    model = MockModel()
    indexer = BrainIndexer(db_path, model=model)
    return indexer

def test_search_functionality(db_path, indexer):
    # 1. Index a dummy file
    dummy_file = db_path.parent / "test_doc.txt"
    dummy_file.write_text("This is a test document about artificial intelligence.", encoding="utf-8")
    
    indexer.index_file(dummy_file)
    
    # 2. Perform search
    model = MockModel()
    results = perform_search("intelligence", model, db_path)
    
    # 3. Verify results
    assert len(results) > 0
    # Result format: (similarity, filename, path, snippet)
    assert results[0][1] == "test_doc.txt"
    assert "test document" in results[0][3]

def test_large_file_indexing_limit(db_path, indexer):
    # Create a large file (e.g., 1MB)
    large_file = db_path.parent / "large_doc.txt"
    # Write 1MB of 'a's
    with open(large_file, "w") as f:
        f.write("a" * 1024 * 1024)
        
    indexer.index_file(large_file)
    
    # Check that we didn't crash and stored it
    conn = duckdb.connect(str(db_path))
    row = conn.execute("SELECT length(full_text) FROM files_index WHERE filename='large_doc.txt'").fetchone()
    conn.close()
    
    # We haven't implemented the limit yet, so this might be 1MB. 
    # After optimization, we want to ensure we don't read/store gigabytes if we only need snippets.
    # For now, just asserting it exists.
    assert row is not None
