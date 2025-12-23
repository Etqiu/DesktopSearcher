## 🧠 Brain Indexer Usage Guide

### Overview
The Brain Indexer automatically watches your Downloads folder (or any target directory) and builds a searchable database of all your files. It extracts text, generates semantic embeddings, and stores everything in DuckDB for fast retrieval.

### Setup & Run

1. **Start the watcher** (runs continuously):
```bash
uv run src/app.py
```

Output:
```
🧠 Brain Indexer starting...
📂 Watching: /Users/you/Downloads
💾 Database: brain.duckdb
📄 Allowed extensions: .pdf, .txt, .md, .docx, .ipynb, .py, .csv

👀 Watching for new files... (Press Ctrl+C to stop)
```

2. **Drop a file** into your Downloads folder:
```
New file detected: paper.pdf
Indexing: paper.pdf
Loading embedding model (one-time setup)...
  ✓ Indexed: paper.pdf (245891 bytes)
```

3. **Search your files semantically**:
```bash
uv run src/query_brain.py
```

Edit the query in `query_brain.py`:
```python
search_files("machine learning algorithms", top_k=5)
```

Output:
```
🔍 Searching for: 'machine learning algorithms'

📊 Top 5 results:

1. deep_learning_paper.pdf (similarity: 0.856)
   Path: /Users/you/Downloads/deep_learning_paper.pdf
   Snippet: Introduction to Neural Networks and Deep Learning...

2. ml_notes.md (similarity: 0.791)
   Path: /Users/you/Downloads/ml_notes.md
   Snippet: ## Supervised Learning Algorithms...
```

### Supported File Types
- **PDFs**: `pypdf` extracts text (first 10 pages)
- **Word docs**: `python-docx` extracts paragraphs
- **Markdown/Text**: Direct read
- **Notebooks**: Extracts code + markdown cells
- **Python/CSV**: Direct read

### Configuration

Edit `WATCH_DIR` in [src/app.py](../src/app.py):
```python
# Watch multiple folders:
WATCH_DIR = Path.home() / "Documents" / "Research"

# Or current directory:
WATCH_DIR = Path.cwd()
```

Add more file extensions:
```python
ALLOWED_EXTENSIONS = {
    ".pdf", ".txt", ".md", ".docx", ".ipynb", ".py", ".csv",
    ".json", ".xml", ".html"  # Add these
}
```

### Architecture

```
┌─────────────┐
│  Downloads  │ ← Filesystem events
└──────┬──────┘
       │
       ▼
┌──────────────┐
│   Watchdog   │ (detects new files)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Text Extract │ (pypdf, docx, etc.)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Embedding   │ (sentence-transformers)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   DuckDB     │ (brain.duckdb)
└──────────────┘
       │
       ▼
┌──────────────┐
│ Query/Search │
└──────────────┘
```

### Database Schema

```sql
CREATE TABLE files_index (
    id INTEGER PRIMARY KEY,
    path VARCHAR,              -- Full file path
    filename VARCHAR,          -- File name only
    extension VARCHAR,         -- .pdf, .txt, etc.
    size_bytes INTEGER,
    created_at TIMESTAMP,
    indexed_at TIMESTAMP,
    text_snippet VARCHAR,      -- First 500 chars
    full_text VARCHAR,         -- Complete extracted text
    embedding FLOAT[384]       -- Semantic vector
);
```

Query directly with DuckDB CLI:
```bash
duckdb brain.duckdb "SELECT filename, size_bytes FROM files_index ORDER BY indexed_at DESC LIMIT 10"
```

### Tips

**Background process** (macOS/Linux):
```bash
nohup uv run src/app.py > watcher.log 2>&1 &
```

**Stop background watcher**:
```bash
pkill -f "src/app.py"
```

**Re-index all files** in Downloads:
```python
from pathlib import Path
from src.app import BrainIndexer

idx = BrainIndexer(Path("brain.duckdb"))
for file in Path.home().glob("Downloads/*"):
    if file.suffix in {".pdf", ".txt", ".md"}:
        idx.index_file(file)
idx.close()
```

**Clear database and start fresh**:
```bash
rm brain.duckdb
uv run src/app.py
```

### Next Steps

1. **Add more extractors**: HTML, JSON, XML parsers
2. **Build a UI**: Streamlit/Gradio search interface
3. **Auto-sort**: Move files based on content
4. **RAG pipeline**: Connect to LLM for Q&A over your files
5. **Multi-folder watch**: Monitor Desktop, Documents, etc.
6. **Duplicate detection**: Find similar files by embedding distance

### Troubleshooting

**"libomp not found" on macOS** (for xgboost/lightgbm):
```bash
brew install libomp
```

**Model download is slow**:
First run downloads ~90MB model from HuggingFace. Be patient or use a smaller model:
```python
self.model = SentenceTransformer('paraphrase-MiniLM-L3-v2')  # Smaller
```

**File not indexed**:
- Check extension is in `ALLOWED_EXTENSIONS`
- Ensure file isn't temporary (`.crdownload`, `.part`)
- Check `watcher.log` for errors

**Out of memory**:
For large PDFs, reduce page limit:
```python
for page in reader.pages[:5]:  # Only first 5 pages
```
