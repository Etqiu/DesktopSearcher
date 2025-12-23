# test-project

A minimal Python starter managed with `uv`, featuring a **Brain Indexer** that watches your Downloads folder, extracts text from files, generates embeddings, and stores everything in DuckDB for semantic search.

## Features

### 🧠 Brain Indexer
- **Filesystem watcher**: Monitors Downloads (or any folder) for new files
- **Text extraction**: PDFs, Word docs, markdown, notebooks, code files
- **Embeddings**: Generates 384-dim vectors via `sentence-transformers`
- **DuckDB storage**: Fast local database for metadata + embeddings
- **Semantic search**: Query your files by meaning, not just keywords
- **🎯 macOS Menu Bar Widget**: Native status bar app for quick search access

#### How it works
```
New file in Downloads → Extract text → Generate embedding → Store in DuckDB → Semantic search
```

**Example:**
```bash
# Start the watcher
$ uv run src/app.py
🧠 Brain Indexer starting...
📂 Watching: ~/Downloads
New file detected: paper.pdf
  ✓ Indexed: paper.pdf (245891 bytes)

# Search semantically
$ uv run src/query_brain.py
🔍 Searching for: 'machine learning'
1. deep_learning_paper.pdf (similarity: 0.856)
2. ml_notes.md (similarity: 0.791)
```

📖 **[Full Documentation](docs/IMPLEMENTATION.md)** | **[Usage Guide](docs/USAGE.md)**

## Quick Start

### 🎨 Launch the Menu Bar Widget (Recommended)
```bash
./launch_widget.sh
```
Or:
```bash
uv run python src/brain_widget.py
```

The **🧠 Brain Widget** appears in your macOS status bar (top-right):
- **Search Files...** - Semantic search with AI-powered ranking
- **Recent Files** - Quick access to last 10 indexed files  
- **Open Database** - View stats, file counts, and storage info
- **Reindex Downloads** - Manually scan and index all files in Downloads

📖 **[Widget Guide](docs/WIDGET.md)** - Auto-start, keyboard shortcuts, customization

**Screenshot:**
```
  🧠                          ← Click this icon
  ┌─────────────────────────┐
  │ Search Files...         │ ← Type query, get AI results
  │ ────────────────────    │
  │ Recent Files        >   │ ← Quick access
  │ ────────────────────    │
  │ Open Database           │
  │ Reindex Downloads       │
  │ ────────────────────    │
  │ Quit                    │
  └─────────────────────────┘
```

### 🤖 Run the Background Watcher (Alternative)
```bash
uv run src/app.py
```
### 🤖 Run the Background Watcher (Alternative)
```bash
uv run src/app.py
```
This will:
- Watch `~/Downloads` for new files (configurable in `app.py`)
- Extract text from supported formats (`.pdf`, `.docx`, `.txt`, `.md`, `.ipynb`, `.py`, `.csv`)
- Generate embeddings and store in `brain.duckdb`
- Keep running until you press `Ctrl+C`

**Use both together:** Run the watcher in the background + use the widget for search!

### 💻 Command-Line Search
```bash
uv run src/query_brain.py
```
Edit the query in `query_brain.py` to search your indexed files by meaning.

📖 **[Full Usage Guide](docs/USAGE.md)** - Background processes, re-indexing, troubleshooting, and more.

### Activate venv (optional)
```bash
source .venv/bin/activate
```

### Run the app
```bash
uv run src/app.py
```

### Add a dependency
```bash
uv add requests
```

### Install data science stack
```bash
uv add numpy pandas scipy scikit-learn matplotlib seaborn statsmodels pyarrow polars xgboost lightgbm
```

### Install common utilities
```bash
uv add requests httpx pydantic python-dotenv tqdm rich loguru sqlalchemy psycopg2-binary typer
```

### Install dev tools
```bash
uv add --group dev pytest ruff black mypy pre-commit
```

### Run a one-off command
```bash
uv run python -c "print('uv is working')"
```

## Notes
- `uv venv` created `.venv` in this folder.
- VS Code is configured to use `.venv/bin/python` automatically.
- **Brain database**: `brain.duckdb` stores all indexed file metadata and embeddings.
- **Watch folder**: Edit `WATCH_DIR` in [src/app.py](src/app.py) to monitor different folders.
- **Embedding model**: First run downloads `all-MiniLM-L6-v2` (~90MB) from HuggingFace.

### Configuration
In [src/app.py](src/app.py):
```python
WATCH_DIR = Path.home() / "Downloads"  # Change to any folder
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx", ".ipynb", ".py", ".csv"}
STABILITY_DELAY = 2  # seconds to wait for file write completion
```

### macOS note: OpenMP for xgboost/lightgbm
- On macOS, `xgboost` and `lightgbm` require the OpenMP runtime (`libomp`).
- Install via Homebrew:
	```bash
	/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
	export PATH="/opt/homebrew/bin:$PATH"
	brew install libomp
	```
 - After installing `libomp`, re-run tests:
	```bash
	uv run -m pytest -q
	```
