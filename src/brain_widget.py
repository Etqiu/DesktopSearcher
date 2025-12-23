"""
Brain Indexer - macOS Menu Bar Widget
A native Mac status bar app for quick semantic search of indexed files.
"""
import rumps
import subprocess
from pathlib import Path
from datetime import datetime
import duckdb
from sentence_transformers import SentenceTransformer
import numpy as np
from AppKit import NSApplication, NSApplicationActivationPolicyAccessory

DB_PATH = Path(__file__).parent.parent / "brain.duckdb"


class BrainWidget(rumps.App):
    """macOS menu bar widget for the Brain Indexer."""
    
    def __init__(self):
        super(BrainWidget, self).__init__(
            "🧠",  # Icon in menu bar
            title="Brain",
            quit_button=None,  # We'll add custom quit
            icon=None,
            template=True
        )
        # Hide dock icon
        NSApplication.sharedApplication().setActivationPolicy_(NSApplicationActivationPolicyAccessory)
        
        self.model = None
        self.menu = [
            "Search Files...",
            None,  # Separator
            "Recent Files",
            None,
            "Open Database",
            "Reindex Downloads",
            None,
            rumps.MenuItem("Quit", callback=self.quit_app)
        ]
        
        # Start with recent files
        self.update_recent_files()
    
    def _get_model(self):
        """Lazy load embedding model."""
        if self.model is None:
            rumps.notification(
                title="Brain Indexer",
                subtitle="Loading AI model...",
                message="First-time setup, please wait.",
                sound=False
            )
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        return self.model
    
    def update_recent_files(self):
        """Update the Recent Files submenu with latest indexed files."""
        try:
            conn = duckdb.connect(str(DB_PATH), read_only=True)
            results = conn.execute("""
                SELECT filename, path, indexed_at 
                FROM files_index 
                ORDER BY indexed_at DESC 
                LIMIT 10
            """).fetchall()
            conn.close()
            
            # Build submenu
            recent_menu = []
            if results:
                for filename, path, indexed_at in results:
                    # Format time
                    if indexed_at:
                        time_str = indexed_at.strftime("%b %d, %H:%M")
                    else:
                        time_str = "Unknown"
                    
                    # Create menu item with callback
                    item = rumps.MenuItem(
                        f"{filename} ({time_str})",
                        callback=lambda sender, p=path: self.open_file(p)
                    )
                    recent_menu.append(item)
            else:
                recent_menu.append(rumps.MenuItem("No files indexed yet", callback=None))
            
            # Replace the "Recent Files" item with submenu
            self.menu["Recent Files"] = recent_menu
            
        except Exception as e:
            print(f"Error loading recent files: {e}")
    
    @rumps.clicked("Search Files...")
    def search_dialog(self, _):
        """Show search dialog."""
        window = rumps.Window(
            message="Enter search query:",
            title="Brain Search",
            default_text="",
            ok="Search",
            cancel="Cancel",
            dimensions=(320, 24)
        )
        
        response = window.run()
        
        if response.clicked:
            query = response.text.strip()
            if query:
                self.perform_search(query)
    
    def perform_search(self, query: str):
        """Execute semantic search and show results."""
        try:
            # Show searching notification
            rumps.notification(
                title="Brain Indexer",
                subtitle=f"Searching: {query}",
                message="Analyzing your files...",
                sound=False
            )
            
            # Load model and generate embedding
            model = self._get_model()
            query_embedding = model.encode(query).tolist()
            
            # Query database
            conn = duckdb.connect(str(DB_PATH), read_only=True)
            results = conn.execute("""
                SELECT filename, path, text_snippet, embedding
                FROM files_index
                WHERE embedding IS NOT NULL
            """).fetchall()
            conn.close()
            
            if not results:
                rumps.alert(
                    title="No Results",
                    message="No indexed files found. Start the watcher to index files.",
                    ok="OK"
                )
                return
            
            # Calculate similarities
            similarities = []
            for filename, path, snippet, embedding in results:
                if embedding:
                    dot_product = np.dot(query_embedding, embedding)
                    norm_query = np.linalg.norm(query_embedding)
                    norm_doc = np.linalg.norm(embedding)
                    similarity = dot_product / (norm_query * norm_doc)
                    similarities.append((similarity, filename, path, snippet))
            
            # Sort by similarity
            similarities.sort(reverse=True, key=lambda x: x[0])
            
            # Show top 5 results
            top_results = similarities[:5]
            
            # Build results message for dialog
            results_text = f"Query: '{query}'\n\n"
            results_text += "Top Results:\n" + "=" * 50 + "\n\n"
            
            for i, (score, filename, path, snippet) in enumerate(top_results, 1):
                score_pct = int(score * 100)
                results_text += f"{i}. {filename} ({score_pct}%)\n"
                results_text += f"   Path: {path}\n"
                results_text += f"   Preview: {snippet[:80]}...\n\n"
            
            # Show results dialog
            response = rumps.alert(
                title="Search Results",
                message=results_text,
                ok="Done",
                cancel=None
            )
            
            # Also show success notification
            rumps.notification(
                title="Search Complete",
                subtitle=f"Found {len(similarities)} files",
                message=f"Top match: {top_results[0][1]} ({int(top_results[0][0]*100)}%)",
                sound=False
            )
            
        except Exception as e:
            rumps.alert(
                title="Search Error",
                message=f"Error performing search: {str(e)}",
                ok="OK"
            )
    
    def open_file(self, path: str):
        """Open file with default application."""
        try:
            subprocess.run(["open", path], check=True)
        except Exception as e:
            rumps.alert(
                title="Error Opening File",
                message=str(e),
                ok="OK"
            )
    
    def reveal_in_finder(self, path: str):
        """Reveal file in Finder."""
        try:
            subprocess.run(["open", "-R", path], check=True)
        except Exception as e:
            rumps.alert(
                title="Error",
                message=str(e),
                ok="OK"
            )
    
    @rumps.clicked("Open Database")
    def open_database(self, _):
        """Show database info and quick stats."""
        try:
            conn = duckdb.connect(str(DB_PATH), read_only=True)
            
            # Get stats
            total_files = conn.execute("SELECT COUNT(*) FROM files_index").fetchone()[0]
            total_size = conn.execute("SELECT SUM(size_bytes) FROM files_index").fetchone()[0] or 0
            total_size_mb = total_size / (1024 * 1024)
            
            # Get file type breakdown
            by_type = conn.execute("""
                SELECT extension, COUNT(*) 
                FROM files_index 
                GROUP BY extension 
                ORDER BY COUNT(*) DESC 
                LIMIT 5
            """).fetchall()
            
            conn.close()
            
            # Build message
            type_breakdown = "\n".join([f"  {ext}: {count}" for ext, count in by_type])
            
            rumps.alert(
                title="Brain Database Stats",
                message=f"Total Files: {total_files}\n"
                        f"Total Size: {total_size_mb:.1f} MB\n\n"
                        f"By Type:\n{type_breakdown}\n\n"
                        f"Location: {DB_PATH}",
                ok="OK"
            )
            
        except Exception as e:
            rumps.alert(
                title="Database Error",
                message=f"Could not read database: {str(e)}\n\n"
                        "Make sure the watcher has indexed some files.",
                ok="OK"
            )
    
    @rumps.clicked("Reindex Downloads")
    def reindex_downloads(self, _):
        """Trigger re-indexing of Downloads folder."""
        response = rumps.alert(
            title="Reindex Downloads?",
            message="This will scan your Downloads folder and index all supported files.\n\n"
                    "This may take a few minutes.",
            ok="Start Indexing",
            cancel="Cancel"
        )
        
        if response == 1:  # OK clicked
            try:
                # Run indexing in background
                from src.app import BrainIndexer
                
                rumps.notification(
                    title="Brain Indexer",
                    subtitle="Indexing started",
                    message="Scanning Downloads folder...",
                    sound=False
                )
                
                # Index all files in Downloads
                indexer = BrainIndexer(DB_PATH)
                downloads = Path.home() / "Downloads"
                
                count = 0
                for file in downloads.glob("*"):
                    if file.is_file() and file.suffix in {".pdf", ".txt", ".md", ".docx", ".ipynb", ".py", ".csv"}:
                        indexer.index_file(file)
                        count += 1
                
                indexer.close()
                
                # Update recent files
                self.update_recent_files()
                
                rumps.notification(
                    title="Indexing Complete",
                    subtitle=f"Indexed {count} files",
                    message="Your Brain is up to date!",
                    sound=True
                )
                
            except Exception as e:
                rumps.alert(
                    title="Indexing Error",
                    message=f"Error during indexing: {str(e)}",
                    ok="OK"
                )
    
    def quit_app(self, _):
        """Quit the application."""
        rumps.quit_application()


if __name__ == "__main__":
    app = BrainWidget()
    app.run()
