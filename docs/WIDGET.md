# 🎨 Brain Widget - macOS Menu Bar App

A native macOS status bar widget for instant semantic search of your indexed files.

## Features

✅ **Always accessible** - Lives in your menu bar  
✅ **Instant search** - Type query, get AI-ranked results  
✅ **Recent files** - Quick access to last 10 indexed items  
✅ **Database stats** - See what's indexed at a glance  
✅ **One-click indexing** - Manually scan Downloads folder  
✅ **File actions** - Open file or reveal in Finder  

## Launch the Widget

### Method 1: Double-click launcher
```bash
./launch_widget.sh
```

### Method 2: Command line
```bash
uv run python src/brain_widget.py
```

### Method 3: Auto-start on login (macOS)

Create a Launch Agent to start the widget automatically:

```bash
cat > ~/Library/LaunchAgents/com.brain.widget.plist <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.brain.widget</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/YOUR_USERNAME/.local/bin/uv</string>
        <string>run</string>
        <string>python</string>
        <string>src/brain_widget.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/Users/YOUR_USERNAME/Documents/test-project</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/brain_widget.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/brain_widget_error.log</string>
</dict>
</plist>
EOF

# Replace YOUR_USERNAME with your actual username
# Then load it:
launchctl load ~/Library/LaunchAgents/com.brain.widget.plist
```

## Usage

### Search Files
1. Click the **🧠** icon in menu bar
2. Select **"Search Files..."**
3. Type your query: `"machine learning papers"`
4. View results ranked by relevance (0-100%)
5. Click result → Open File or Reveal in Finder

### Recent Files
- Click **"Recent Files"** to see last 10 indexed items
- Shows filename + timestamp
- Click to open instantly

### Database Stats
- Click **"Open Database"** to see:
  - Total files indexed
  - Total storage size
  - Breakdown by file type (.pdf, .txt, etc.)
  - Database location

### Manual Reindex
- Click **"Reindex Downloads"**
- Scans entire Downloads folder
- Indexes all supported files
- Shows notification when complete

## How Search Works

```
Your Query: "python data visualization"
       ↓
AI Embedding (sentence-transformers)
       ↓
Compare with all indexed files
       ↓
Rank by cosine similarity (0-1)
       ↓
Display top 5 matches with scores
```

**Example Results:**
```
1. plotly_tutorial.ipynb (87%)
   → Open File / Reveal in Finder
   Preview: "Data visualization with plotly..."

2. matplotlib_guide.pdf (72%)
   → Open File / Reveal in Finder
   Preview: "Introduction to matplotlib..."

3. seaborn_examples.py (64%)
   ...
```

## Menu Structure

```
🧠 Brain
├── Search Files...          ← Main search
├── ────────────────
├── Recent Files             ← Quick access
│   ├── report.pdf (Dec 22, 14:30)
│   ├── notes.md (Dec 22, 13:15)
│   └── ...
├── ────────────────
├── Open Database            ← Stats & info
├── Reindex Downloads        ← Manual scan
├── ────────────────
└── Quit
```

## Notifications

The widget sends macOS notifications for:
- **Search started**: "Analyzing your files..."
- **Search complete**: "Found 15 files, top match: paper.pdf (89%)"
- **Indexing started**: "Scanning Downloads folder..."
- **Indexing complete**: "Indexed 12 files"

## Tips

**Search Tips:**
- Use natural language: "machine learning tutorials"
- Be specific: "python asyncio documentation"
- Semantic matching works: "ML" finds "machine learning" files

**Performance:**
- First search loads AI model (~5 sec, one-time)
- Subsequent searches are instant (<500ms)
- 1000 files = ~1.5 MB database

**Keyboard Shortcuts:**
- `Cmd+Space` → Type "brain" → Enter (if using Spotlight)
- Or assign global hotkey via macOS Keyboard settings

## Troubleshooting

**Widget doesn't appear:**
- Check Activity Monitor for `python brain_widget.py`
- View logs: `tail -f /tmp/brain_widget.log`

**"No indexed files found":**
- Run the watcher first: `uv run src/app.py`
- Or use "Reindex Downloads" from menu

**Search is slow:**
- First search loads model (normal)
- Check CPU usage in Activity Monitor
- Reduce indexed file count if needed

**Widget crashes:**
```bash
# View error logs
cat /tmp/brain_widget_error.log

# Restart
pkill -f brain_widget
./launch_widget.sh
```

## Advanced: Customize the Widget

Edit [src/brain_widget.py](../src/brain_widget.py):

**Change icon:**
```python
super(BrainWidget, self).__init__(
    "🔍",  # Change to any emoji or text
    ...
)
```

**Add more menu items:**
```python
self.menu = [
    "Search Files...",
    "Advanced Search...",  # Add new feature
    "Settings...",         # Add settings
    None,
    ...
]
```

**Customize search results count:**
```python
top_results = similarities[:10]  # Show top 10 instead of 5
```

## Widget + Watcher Together

**Recommended setup:**
1. Start watcher in background:
   ```bash
   nohup uv run src/app.py > watcher.log 2>&1 &
   ```

2. Launch widget:
   ```bash
   ./launch_widget.sh
   ```

Now you have:
- **Automatic indexing** (watcher monitors Downloads)
- **Instant search** (widget in menu bar)
- **Zero friction** - works in the background!

## Uninstall

Remove auto-start:
```bash
launchctl unload ~/Library/LaunchAgents/com.brain.widget.plist
rm ~/Library/LaunchAgents/com.brain.widget.plist
```

Stop running widget:
```bash
pkill -f brain_widget
```

---

**Built with:** [rumps](https://github.com/jaredks/rumps) - Ridiculously Uncomplicated macOS Python Statusbar apps
