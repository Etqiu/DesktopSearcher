"""
Filesystem watcher that monitors Downloads folder and indexes files into DuckDB.
Extracts text, generates embeddings, and stores metadata for later retrieval.
"""
import os
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
import duckdb
from sentence_transformers import SentenceTransformer

# Text extraction libraries
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None


# Configuration
WATCH_DIR = Path.home() / "Downloads"  # Change to any folder
DB_PATH = Path(__file__).parent.parent / "brain.duckdb"
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx", ".ipynb", ".py", ".csv"}
TEMP_EXTENSIONS = {".crdownload", ".part", ".tmp", ".download"}
STABILITY_DELAY = 2  # seconds to wait for file to finish writing


class BrainIndexer:
    """Handles text extraction, embedding generation, and DB storage."""
    
    def __init__(self, db_path: Path, model=None):
        self.db_path = db_path
        self.conn = duckdb.connect(str(db_path))
        self.model = model  # Use provided model or lazy load
        self._init_db()
    
    def _init_db(self):
        """Create the files index table if it doesn't exist."""
        # Create sequence first
        self.conn.execute("CREATE SEQUENCE IF NOT EXISTS files_index_seq START 1")
        
        # Create table with auto-incrementing ID
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS files_index (
                id INTEGER PRIMARY KEY DEFAULT nextval('files_index_seq'),
                path VARCHAR,
                filename VARCHAR,
                extension VARCHAR,
                size_bytes INTEGER,
                created_at TIMESTAMP,
                indexed_at TIMESTAMP,
                text_snippet VARCHAR,
                full_text VARCHAR,
                embedding FLOAT[384]
            )
        """)
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_path ON files_index(path)")
    
    def _get_embedding_model(self):
        """Lazy load the embedding model (first call takes time)."""
        if self.model is None:
            print("Loading embedding model (one-time setup)...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dim embeddings
        return self.model
    
    def extract_text(self, file_path: Path) -> Optional[str]:
        """Extract text from supported file types."""
        ext = file_path.suffix.lower()
        
        try:
            if ext == ".txt" or ext == ".md" or ext == ".py" or ext == ".csv":
                return file_path.read_text(encoding="utf-8", errors="ignore")
            
            elif ext == ".pdf" and PdfReader:
                reader = PdfReader(str(file_path))
                text = []
                for page in reader.pages[:10]:  # Limit to first 10 pages
                    text.append(page.extract_text())
                return "\n".join(text)
            
            elif ext == ".docx" and DocxDocument:
                doc = DocxDocument(str(file_path))
                return "\n".join([para.text for para in doc.paragraphs])
            
            elif ext == ".ipynb":
                import json
                notebook = json.loads(file_path.read_text())
                cells = notebook.get("cells", [])
                text = []
                for cell in cells:
                    if cell.get("cell_type") in ["markdown", "code"]:
                        source = cell.get("source", [])
                        if isinstance(source, list):
                            text.append("".join(source))
                        else:
                            text.append(source)
                return "\n".join(text)
            
        except Exception as e:
            print(f"Error extracting text from {file_path}: {e}")
            return None
        
        return None
    
    def index_file(self, file_path: Path):
        """Process a file: extract text, generate embedding, store in DB."""
        if not file_path.exists():
            return
        
        # Get metadata
        stat = file_path.stat()
        try:
            # On macOS/BSD, st_birthtime is the creation (download) time
            created_at = datetime.fromtimestamp(stat.st_birthtime)
        except AttributeError:
            # Fallback for Linux/Windows
            created_at = datetime.fromtimestamp(stat.st_ctime)
            
        size_bytes = stat.st_size
        
        # Extract text
        print(f"Indexing: {file_path.name}")
        full_text = self.extract_text(file_path)
        
        if not full_text:
            print(f"  → No text extracted, skipping embedding")
            snippet = ""
            embedding = None
        else:
            snippet = full_text[:2000]  # First 2000 chars
            
            # Generate embedding
            model = self._get_embedding_model()
            embedding = model.encode(full_text[:10000], show_progress_bar=False)  # Limit text length
            embedding = embedding.tolist()
        
        # Insert into DB
        self.conn.execute("""
            INSERT INTO files_index (
                path, filename, extension, size_bytes, 
                created_at, indexed_at, text_snippet, full_text, embedding
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            str(file_path.absolute()),
            file_path.name,
            file_path.suffix,
            size_bytes,
            created_at,
            datetime.now(),
            snippet,
            full_text,
            embedding
        ])
        self.conn.commit()
        print(f"  ✓ Indexed: {file_path.name} ({size_bytes} bytes)")
    
    def close(self):
        """Close the database connection."""
        self.conn.close()


class DownloadWatcherHandler(FileSystemEventHandler):
    """Handles filesystem events and triggers indexing."""
    
    def __init__(self, indexer: BrainIndexer):
        self.indexer = indexer
        self.processing = set()  # Track files being processed
    
    def on_created(self, event):
        """Called when a file or directory is created."""
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        
        # Ignore temporary files
        if file_path.suffix.lower() in TEMP_EXTENSIONS:
            return
        
        # Ignore hidden files
        if file_path.name.startswith("."):
            return
        
        # Only process allowed extensions
        if file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
            return
        
        # Avoid duplicate processing
        if str(file_path) in self.processing:
            return
        
        self.processing.add(str(file_path))
        
        # Wait for file to stabilize (finish writing)
        print(f"New file detected: {file_path.name}")
        time.sleep(STABILITY_DELAY)
        
        # Check if file still exists and size is stable
        if not file_path.exists():
            self.processing.discard(str(file_path))
            return
        
        try:
            initial_size = file_path.stat().st_size
            time.sleep(0.5)
            final_size = file_path.stat().st_size
            
            if initial_size != final_size:
                print(f"  → File still writing, skipping for now")
                self.processing.discard(str(file_path))
                return
            
            # Process the file
            self.indexer.index_file(file_path)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
        finally:
            self.processing.discard(str(file_path))


def main():
    """Start the filesystem watcher."""
    print(f"🧠 Brain Indexer starting...")
    print(f"📂 Watching: {WATCH_DIR}")
    print(f"💾 Database: {DB_PATH}")
    print(f"📄 Allowed extensions: {', '.join(ALLOWED_EXTENSIONS)}")
    print()
    
    # Ensure watch directory exists
    if not WATCH_DIR.exists():
        print(f"Error: Watch directory does not exist: {WATCH_DIR}")
        print("Update WATCH_DIR in the script to point to an existing folder.")
        return
    
    # Initialize indexer
    indexer = BrainIndexer(DB_PATH)
    
    # Set up filesystem observer
    event_handler = DownloadWatcherHandler(indexer)
    observer = Observer()
    observer.schedule(event_handler, str(WATCH_DIR), recursive=False)
    observer.start()
    
    print("👀 Watching for new files... (Press Ctrl+C to stop)\n")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping watcher...")
        observer.stop()
        observer.join()
        indexer.close()
        print("✓ Shutdown complete")


if __name__ == "__main__":
    main()
