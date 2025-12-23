"""
Quick demo of the Brain Widget functionality.
Run this to test all features without launching the GUI.
"""
from pathlib import Path
import duckdb
from src.app import BrainIndexer

def demo():
    print("=" * 60)
    print("🧠 BRAIN WIDGET DEMO")
    print("=" * 60)
    
    # Initialize
    db_path = Path("brain.duckdb")
    
    # Show database stats
    print("\n📊 DATABASE STATS:")
    try:
        conn = duckdb.connect(str(db_path), read_only=True)
        
        total_files = conn.execute("SELECT COUNT(*) FROM files_index").fetchone()[0]
        total_size = conn.execute("SELECT SUM(size_bytes) FROM files_index").fetchone()[0] or 0
        total_size_mb = total_size / (1024 * 1024)
        
        print(f"  Total Files: {total_files}")
        print(f"  Total Size: {total_size_mb:.2f} MB")
        
        # By type
        print("\n  By Type:")
        by_type = conn.execute("""
            SELECT extension, COUNT(*) 
            FROM files_index 
            GROUP BY extension 
            ORDER BY COUNT(*) DESC
        """).fetchall()
        
        for ext, count in by_type:
            print(f"    {ext}: {count}")
        
        # Recent files
        print("\n📂 RECENT FILES (Last 5):")
        recent = conn.execute("""
            SELECT filename, indexed_at 
            FROM files_index 
            ORDER BY indexed_at DESC 
            LIMIT 5
        """).fetchall()
        
        for filename, indexed_at in recent:
            time_str = indexed_at.strftime("%b %d, %H:%M") if indexed_at else "Unknown"
            print(f"  • {filename} ({time_str})")
        
        conn.close()
        
    except Exception as e:
        print(f"  ⚠️  Error: {e}")
        print("  Run the watcher first to index some files!")
    
    print("\n" + "=" * 60)
    print("🎨 To launch the actual widget, run:")
    print("   ./launch_widget.sh")
    print("=" * 60)

if __name__ == "__main__":
    demo()
