import sys
import os
import threading
import time
import subprocess
import webbrowser
import socket
import duckdb
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import objc
from AppKit import (
    NSApplication, NSWindow, NSWindowStyleMaskTitled, NSWindowStyleMaskClosable,
    NSWindowStyleMaskResizable, NSBackingStoreBuffered, NSColor, NSRect, NSPoint, NSSize,
    NSTextField, NSButton, NSStackView, NSScrollView, NSView, NSVisualEffectView,
    NSVisualEffectMaterialHUDWindow, NSVisualEffectBlendingModeBehindWindow,
    NSLayoutAttributeCenterX, NSLayoutAttributeCenterY, NSLayoutAttributeWidth,
    NSLayoutAttributeHeight, NSLayoutAttributeTop, NSLayoutAttributeLeading,
    NSLayoutAttributeTrailing, NSLayoutAttributeBottom, NSLayoutConstraint,
    NSApplicationActivationPolicyRegular, NSApplicationActivationPolicyAccessory, NSObject, NSFont, NSBezelStyleRounded,
    NSTextFieldSquareBezel, NSFocusRingTypeNone, NSBox, NSBoxCustom, NSBoxSeparator,
    NSCursor, NSTrackingArea, NSTrackingMouseEnteredAndExited, NSTrackingActiveInActiveApp,
    NSWindowStyleMaskFullSizeContentView, NSWindowTitleHidden, NSButtonTypeMomentaryPushIn,
    NSWorkspace
)
from Foundation import NSMakeRect, NSTimer, NSURL, NSURLRequest
from WebKit import WKWebView

from app import DB_PATH, BrainIndexer, WATCH_DIR
from brain_search import perform_search

# --- UI Constants ---
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 500
ROW_HEIGHT = 50

class SearchResultRow(NSView):
    """Custom view for a result row."""
    def initWithResult_callback_(self, result, callback):
        self = objc.super(SearchResultRow, self).init()
        if self:
            self.result = result
            self.callback = callback
            self.path = result[2]
            
            # Setup layout
            self.setTranslatesAutoresizingMaskIntoConstraints_(False)
            
            # Filename Label
            score_pct = int(result[0] * 100)
            title = f"{result[1]} ({score_pct}%)"
            
            self.titleLabel = NSTextField.labelWithString_(title)
            self.titleLabel.setFont_(NSFont.systemFontOfSize_(14))
            self.titleLabel.setTextColor_(NSColor.whiteColor())
            self.titleLabel.setTranslatesAutoresizingMaskIntoConstraints_(False)
            self.addSubview_(self.titleLabel)
            
            # Path Label (smaller)
            path_str = str(self.path)
            if len(path_str) > 60:
                path_str = "..." + path_str[-57:]
            self.pathLabel = NSTextField.labelWithString_(path_str)
            self.pathLabel.setFont_(NSFont.systemFontOfSize_(11))
            self.pathLabel.setTextColor_(NSColor.lightGrayColor())
            self.pathLabel.setTranslatesAutoresizingMaskIntoConstraints_(False)
            self.addSubview_(self.pathLabel)
            
            # Clickable button (invisible overlay)
            self.btn = NSButton.alloc().init()
            self.btn.setTitle_("")
            self.btn.setTransparent_(True)
            self.btn.setTranslatesAutoresizingMaskIntoConstraints_(False)
            self.btn.setTarget_(self)
            self.btn.setAction_("clicked:")
            self.addSubview_(self.btn)
            
            # Constraints
            NSLayoutConstraint.activateConstraints_([
                self.titleLabel.topAnchor().constraintEqualToAnchor_constant_(self.topAnchor(), 5),
                self.titleLabel.leadingAnchor().constraintEqualToAnchor_constant_(self.leadingAnchor(), 10),
                self.titleLabel.trailingAnchor().constraintEqualToAnchor_constant_(self.trailingAnchor(), -10),
                
                self.pathLabel.topAnchor().constraintEqualToAnchor_(self.titleLabel.bottomAnchor()),
                self.pathLabel.leadingAnchor().constraintEqualToAnchor_constant_(self.leadingAnchor(), 10),
                self.pathLabel.trailingAnchor().constraintEqualToAnchor_constant_(self.trailingAnchor(), -10),
                
                self.btn.topAnchor().constraintEqualToAnchor_(self.topAnchor()),
                self.btn.bottomAnchor().constraintEqualToAnchor_(self.bottomAnchor()),
                self.btn.leadingAnchor().constraintEqualToAnchor_(self.leadingAnchor()),
                self.btn.trailingAnchor().constraintEqualToAnchor_(self.trailingAnchor()),
                
                self.heightAnchor().constraintEqualToConstant_(ROW_HEIGHT)
            ])
            
        return self

    def clicked_(self, sender):
        try:
            if self.callback:
                self.callback(self.path)
        except Exception as e:
            print(f"Click error: {e}")

class BrainApp(NSObject):
    def applicationShouldTerminateAfterLastWindowClosed_(self, sender):
        return True

    def applicationDidFinishLaunching_(self, notification):
        self.model = None
        self.search_timer = None
        self.is_indexing = False
        
        # Auto-reindex on startup
        threading.Thread(target=self.reindex_thread, daemon=True).start()
        
        # Create Window
        self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT),
            NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskResizable | NSWindowStyleMaskFullSizeContentView,
            NSBackingStoreBuffered,
            False
        )
        self.window.center()
        self.window.setTitle_("Brain Search")
        self.window.setTitlebarAppearsTransparent_(True)
        self.window.setTitleVisibility_(NSWindowTitleHidden)
        self.window.setOpaque_(False)
        self.window.setBackgroundColor_(NSColor.clearColor())
        self.window.setLevel_(3) # Floating level
        self.window.setMovableByWindowBackground_(True)
        
        # Visual Effect View (Blur Background)
        self.visualEffectView = NSVisualEffectView.alloc().initWithFrame_(self.window.contentView().bounds())
        self.visualEffectView.setMaterial_(NSVisualEffectMaterialHUDWindow)
        self.visualEffectView.setBlendingMode_(NSVisualEffectBlendingModeBehindWindow)
        self.visualEffectView.setState_(1) # Active
        self.visualEffectView.setAutoresizingMask_(18) # Width | Height
        self.window.setContentView_(self.visualEffectView)
        
        # Setup UI
        self.setupUI()
        
        self.window.makeKeyAndOrderFront_(None)
        self.window.orderFrontRegardless()
        
        # Load Model
        threading.Thread(target=self.load_model, daemon=True).start()

    def setupUI(self):
        # --- Navigation Bar ---
        self.navStackView = NSStackView.alloc().init()
        self.navStackView.setOrientation_(0) # Horizontal
        self.navStackView.setAlignment_(NSLayoutAttributeCenterY)
        self.navStackView.setSpacing_(20)
        self.navStackView.setTranslatesAutoresizingMaskIntoConstraints_(False)
        self.visualEffectView.addSubview_(self.navStackView)

        # Create Buttons
        self.btnFileSearch = self.createNavButton_action_("File Search", "showFileSearch:")
        self.btnChat = self.createNavButton_action_("Chat", "showChat:")
        self.btnAnalysis = self.createNavButton_action_("Analysis", "showAnalysis:")
        self.btnOrganize = self.createNavButton_action_("Organize", "showOrganize:")
        
        # Highlight "File Search" as active
        self.btnFileSearch.setContentTintColor_(NSColor.whiteColor())

        self.navStackView.addArrangedSubview_(self.btnFileSearch)
        self.navStackView.addArrangedSubview_(self.btnChat)
        self.navStackView.addArrangedSubview_(self.btnAnalysis)
        self.navStackView.addArrangedSubview_(self.btnOrganize)

        # --- Search Field ---
        self.searchField = NSTextField.alloc().init()
        self.searchField.setTranslatesAutoresizingMaskIntoConstraints_(False)
        self.searchField.setPlaceholderString_("Search your brain...")
        self.searchField.setFont_(NSFont.systemFontOfSize_(20))
        self.searchField.setBezeled_(False)
        self.searchField.setDrawsBackground_(False)
        self.searchField.setFocusRingType_(NSFocusRingTypeNone)
        self.searchField.setTextColor_(NSColor.whiteColor())
        self.searchField.setDelegate_(self)
        
        self.visualEffectView.addSubview_(self.searchField)

        # Reindex Button
        self.reindexBtn = NSButton.alloc().init()
        self.reindexBtn.setTitle_("⟳")
        self.reindexBtn.setBezelStyle_(NSBezelStyleRounded)
        self.reindexBtn.setTranslatesAutoresizingMaskIntoConstraints_(False)
        self.reindexBtn.setTarget_(self)
        self.reindexBtn.setAction_("reindex:")
        self.visualEffectView.addSubview_(self.reindexBtn)
        
        # Separator
        self.separator = NSBox.alloc().init()
        self.separator.setBoxType_(NSBoxSeparator)
        self.separator.setTranslatesAutoresizingMaskIntoConstraints_(False)
        self.visualEffectView.addSubview_(self.separator)
        
        # Scroll View for Results
        self.scrollView = NSScrollView.alloc().init()
        self.scrollView.setTranslatesAutoresizingMaskIntoConstraints_(False)
        self.scrollView.setDrawsBackground_(False)
        self.scrollView.setHasVerticalScroller_(True)
        
        # Stack View inside Scroll View
        self.stackView = NSStackView.alloc().init()
        self.stackView.setOrientation_(1) # Vertical
        self.stackView.setAlignment_(NSLayoutAttributeLeading)
        self.stackView.setSpacing_(5)
        self.stackView.setTranslatesAutoresizingMaskIntoConstraints_(False)
        
        # Flip coordinate system for stack view (top-down)
        self.documentView = NSView.alloc().init()
        self.documentView.setTranslatesAutoresizingMaskIntoConstraints_(False)
        self.documentView.addSubview_(self.stackView)
        
        self.scrollView.setDocumentView_(self.documentView)
        self.visualEffectView.addSubview_(self.scrollView)
        
        # --- Analysis View ---
        self.analysisContainer = NSView.alloc().init()
        self.analysisContainer.setTranslatesAutoresizingMaskIntoConstraints_(False)
        self.analysisContainer.setHidden_(True)
        self.visualEffectView.addSubview_(self.analysisContainer)
        
        # Web View
        self.webView = WKWebView.alloc().init()
        self.webView.setTranslatesAutoresizingMaskIntoConstraints_(False)
        # self.webView.setDrawsBackground_(False) # Might cause issues with WKWebView
        self.webView.setValue_forKey_(False, "drawsBackground") # Correct way for WKWebView
        self.analysisContainer.addSubview_(self.webView)
        
        # Constraints
        NSLayoutConstraint.activateConstraints_([
            # Nav Bar
            self.navStackView.topAnchor().constraintEqualToAnchor_constant_(self.visualEffectView.topAnchor(), 15),
            self.navStackView.centerXAnchor().constraintEqualToAnchor_(self.visualEffectView.centerXAnchor()),
            self.navStackView.heightAnchor().constraintEqualToConstant_(30),

            # Search Field (Moved down below Nav Bar)
            self.searchField.topAnchor().constraintEqualToAnchor_constant_(self.navStackView.bottomAnchor(), 20),
            self.searchField.leadingAnchor().constraintEqualToAnchor_constant_(self.visualEffectView.leadingAnchor(), 20),
            self.searchField.trailingAnchor().constraintEqualToAnchor_constant_(self.visualEffectView.trailingAnchor(), -60),
            self.searchField.heightAnchor().constraintEqualToConstant_(40),
            
            # Reindex Button
            self.reindexBtn.centerYAnchor().constraintEqualToAnchor_(self.searchField.centerYAnchor()),
            self.reindexBtn.leadingAnchor().constraintEqualToAnchor_constant_(self.searchField.trailingAnchor(), 10),
            self.reindexBtn.trailingAnchor().constraintEqualToAnchor_constant_(self.visualEffectView.trailingAnchor(), -20),
            self.reindexBtn.heightAnchor().constraintEqualToConstant_(30),

            # Separator
            self.separator.topAnchor().constraintEqualToAnchor_constant_(self.searchField.bottomAnchor(), 10),
            self.separator.leadingAnchor().constraintEqualToAnchor_(self.visualEffectView.leadingAnchor()),
            self.separator.trailingAnchor().constraintEqualToAnchor_(self.visualEffectView.trailingAnchor()),
            self.separator.heightAnchor().constraintEqualToConstant_(1),
            
            # Scroll View
            self.scrollView.topAnchor().constraintEqualToAnchor_constant_(self.separator.bottomAnchor(), 0),
            self.scrollView.leadingAnchor().constraintEqualToAnchor_(self.visualEffectView.leadingAnchor()),
            self.scrollView.trailingAnchor().constraintEqualToAnchor_(self.visualEffectView.trailingAnchor()),
            self.scrollView.bottomAnchor().constraintEqualToAnchor_(self.visualEffectView.bottomAnchor()),
            
            # Stack View
            self.stackView.topAnchor().constraintEqualToAnchor_(self.documentView.topAnchor()),
            self.stackView.leadingAnchor().constraintEqualToAnchor_(self.documentView.leadingAnchor()),
            self.stackView.trailingAnchor().constraintEqualToAnchor_(self.documentView.trailingAnchor()),
            
            self.documentView.widthAnchor().constraintEqualToAnchor_(self.scrollView.widthAnchor()),
            self.documentView.heightAnchor().constraintGreaterThanOrEqualToAnchor_(self.scrollView.heightAnchor()),
            
            # Analysis Container
            self.analysisContainer.topAnchor().constraintEqualToAnchor_constant_(self.navStackView.bottomAnchor(), 20),
            self.analysisContainer.leadingAnchor().constraintEqualToAnchor_(self.visualEffectView.leadingAnchor()),
            self.analysisContainer.trailingAnchor().constraintEqualToAnchor_(self.visualEffectView.trailingAnchor()),
            self.analysisContainer.bottomAnchor().constraintEqualToAnchor_(self.visualEffectView.bottomAnchor()),
            
            # Web View Constraints
            self.webView.topAnchor().constraintEqualToAnchor_(self.analysisContainer.topAnchor()),
            self.webView.leadingAnchor().constraintEqualToAnchor_(self.analysisContainer.leadingAnchor()),
            self.webView.trailingAnchor().constraintEqualToAnchor_(self.analysisContainer.trailingAnchor()),
            self.webView.bottomAnchor().constraintEqualToAnchor_(self.analysisContainer.bottomAnchor())
        ])
        
        # Constraint for document view height to match stack view (dynamic)
        # We want docHeight >= stackHeight AND docHeight >= scrollViewHeight
        # But priority is stackHeight if it's larger.
        # Actually, if we just pin stackView to top, and let documentView be big enough.
        
        self.stackBottomConstraint = self.documentView.bottomAnchor().constraintEqualToAnchor_(self.stackView.bottomAnchor())
        self.stackBottomConstraint.setPriority_(250) # Low priority, allows expansion
        self.stackBottomConstraint.setActive_(True)

    def createNavButton_action_(self, title, action):
        btn = NSButton.alloc().init()
        btn.setTitle_(title)
        btn.setButtonType_(NSButtonTypeMomentaryPushIn)
        btn.setBordered_(False)
        btn.setFont_(NSFont.systemFontOfSize_(13))
        btn.setContentTintColor_(NSColor.lightGrayColor())
        btn.setTarget_(self)
        btn.setAction_(action)
        return btn

    @objc.python_method
    def setFileSearchVisible_(self, visible):
        self.searchField.setHidden_(not visible)
        self.reindexBtn.setHidden_(not visible)
        self.separator.setHidden_(not visible)
        self.scrollView.setHidden_(not visible)

    def setAnalysisVisible_(self, visible):
        if hasattr(self, 'analysisContainer'):
            self.analysisContainer.setHidden_(not visible)

    def openAnalysisBrowser_(self, sender):
        webbrowser.open("http://localhost:8501")

    def showFileSearch_(self, sender):
        print("Switching to File Search")
        self.setFileSearchVisible_(True)
        self.setAnalysisVisible_(False)
        
        # Update Nav Colors
        self.btnFileSearch.setContentTintColor_(NSColor.whiteColor())
        self.btnAnalysis.setContentTintColor_(NSColor.lightGrayColor())

    def showChat_(self, sender):
        print("Switching to Chat (Not Implemented)")

    def showAnalysis_(self, sender):
        print("Switching to Analysis")
        self.setFileSearchVisible_(False)
        self.setAnalysisVisible_(True)
        
        # Update Nav Colors
        self.btnFileSearch.setContentTintColor_(NSColor.lightGrayColor())
        self.btnAnalysis.setContentTintColor_(NSColor.whiteColor())
        
        if not hasattr(self, 'analysis_loaded') or not self.analysis_loaded:
            threading.Thread(target=self._launch_analysis_thread, daemon=True).start()

    @objc.python_method
    def _launch_analysis_thread(self):
        if hasattr(self, 'analysis_loading') and self.analysis_loading:
            return
        self.analysis_loading = True

        # Check if Streamlit is already running on port 8501
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', 8501))
        sock.close()
        
        if result != 0:
            # Port is closed, start Streamlit
            print("Starting Streamlit server...")
            script_path = os.path.join(os.path.dirname(__file__), "brain_analytics.py")
            
            # Log output to file for debugging
            log_file = open(os.path.join(os.path.dirname(__file__), "streamlit.log"), "w")
            
            self.streamlit_process = subprocess.Popen(
                [sys.executable, "-m", "streamlit", "run", script_path, "--server.headless=true"],
                stdout=log_file,
                stderr=log_file
            )
            
            # Wait for it to start (up to 20 seconds)
            started = False
            for i in range(20):
                time.sleep(1)
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                if sock.connect_ex(('localhost', 8501)) == 0:
                    sock.close()
                    started = True
                    print(f"Streamlit started on attempt {i+1}")
                    break
                sock.close()
                print(f"Waiting for Streamlit... {i+1}")
            
            if not started:
                print("Streamlit failed to start within timeout.")
            
        # Load URL in WebView on main thread
        print("Loading URL in WebView...")
        self.performSelectorOnMainThread_withObject_waitUntilDone_("loadAnalysisURL:", "http://localhost:8501", False)
        self.analysis_loaded = True
        self.analysis_loading = False

    def loadAnalysisURL_(self, url_str):
        print(f"Loading: {url_str}")
        if hasattr(self, 'webView'):
            url = NSURL.URLWithString_(url_str)
            req = NSURLRequest.requestWithURL_(url)
            self.webView.loadRequest_(req)
            print("Request sent to WebView")

    def showOrganize_(self, sender):
        print("Switching to Organize (Not Implemented)")

    def reindex_(self, sender):
        if self.is_indexing:
            print("Already indexing...")
            return
        threading.Thread(target=self.reindex_thread, daemon=True).start()

    @objc.python_method
    def reindex_thread(self):
        if self.is_indexing:
            return
        self.is_indexing = True
        
        try:
            self.performSelectorOnMainThread_withObject_waitUntilDone_("updateStatus:", "Checking files...", False)
            
            # Pass existing model to avoid reloading it (saves RAM/CPU)
            indexer = BrainIndexer(DB_PATH, model=self.model)
            
            # Get existing paths to avoid re-indexing
            existing_paths = set()
            try:
                rows = indexer.conn.execute("SELECT path FROM files_index").fetchall()
                for r in rows:
                    existing_paths.add(str(Path(r[0]).absolute()))
            except Exception:
                pass # Table might not exist yet
            
            watch_dir = Path(WATCH_DIR)
            
            count = 0
            extensions = {".pdf", ".txt", ".md", ".docx", ".ipynb", ".py", ".csv"}
            
            if watch_dir.exists():
                for file in watch_dir.glob("*"):
                    if file.is_file() and file.suffix.lower() in extensions:
                        file_path = str(file.absolute())
                        if file_path not in existing_paths:
                            # Throttle UI updates
                            if count % 5 == 0:
                                self.performSelectorOnMainThread_withObject_waitUntilDone_("updateStatus:", f"Indexing {file.name}...", False)
                            indexer.index_file(file)
                            count += 1
                            time.sleep(0.05) # Yield to prevent UI starvation
            
            indexer.close()
            
            if count > 0:
                print(f"Indexed {count} new files")
                self.performSelectorOnMainThread_withObject_waitUntilDone_("updateStatus:", f"Added {count} new files", False)
                time.sleep(2)
            else:
                print("No new files found")
            
            self.performSelectorOnMainThread_withObject_waitUntilDone_("updateStatus:", "Search your brain...", False)
            
        except Exception as e:
            print(f"Reindexing error: {e}")
            self.performSelectorOnMainThread_withObject_waitUntilDone_("updateStatus:", "Indexing failed", False)
        finally:
            self.is_indexing = False

    def updateStatus_(self, text):
        self.searchField.setPlaceholderString_(text)

    def controlTextDidChange_(self, notification):
        query = self.searchField.stringValue()
        
        if self.search_timer:
            self.search_timer.invalidate()
            self.search_timer = None
        
        # Debounce with NSTimer
        self.search_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            0.3, self, "triggerSearch:", query, False
        )

    def triggerSearch_(self, timer):
        query = timer.userInfo()
        self.performSearch_(query)

    def performSearch_(self, query):
        if not query or len(query) < 2:
            self.clearResults()
            return
            
        if not self.model:
            return
            
        threading.Thread(target=self._search_thread, args=(query,), daemon=True).start()

    @objc.python_method
    def _search_thread(self, query):
        try:
            results = perform_search(query, self.model, DB_PATH)
            self.performSelectorOnMainThread_withObject_waitUntilDone_("updateResults:", results, False)
            
        except Exception as e:
            print(f"Search error: {e}")

    def updateResults_(self, results):
        self.clearResults()
        
        for res in results:
            row = SearchResultRow.alloc().initWithResult_callback_(res, self.revealInFinder)
            self.stackView.addArrangedSubview_(row)
            
            # Width constraint
            row.widthAnchor().constraintEqualToAnchor_(self.stackView.widthAnchor()).setActive_(True)

    @objc.python_method
    def clearResults(self):
        for view in self.stackView.arrangedSubviews():
            self.stackView.removeView_(view)
            view.removeFromSuperview()

    @objc.python_method
    def revealInFinder(self, path):
        try:
            ws = NSWorkspace.sharedWorkspace()
            ws.selectFile_inFileViewerRootedAtPath_(path, None)
        except Exception as e:
            print(f"Error: {e}")

    def applicationWillTerminate_(self, notification):
        if hasattr(self, 'streamlit_process') and self.streamlit_process:
            self.streamlit_process.terminate()

    @objc.python_method
    def load_model(self):
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Model loaded")
        except Exception as e:
            print(f"Error loading model: {e}")

if __name__ == "__main__":
    app = NSApplication.sharedApplication()
    delegate = BrainApp.alloc().init()
    app.setDelegate_(delegate)
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    app.activateIgnoringOtherApps_(True)
    app.run()
