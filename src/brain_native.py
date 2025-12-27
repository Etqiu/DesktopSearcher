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
from datetime import datetime
from AppKit import (
    NSApplication, NSWindow, NSWindowStyleMaskTitled, NSWindowStyleMaskClosable,
    NSWindowStyleMaskResizable, NSBackingStoreBuffered, NSColor, NSRect, NSPoint, NSSize,
    NSTextField, NSButton, NSStackView, NSScrollView, NSView, NSVisualEffectView, NSSplitView,
    NSVisualEffectMaterialHUDWindow, NSVisualEffectBlendingModeBehindWindow,
    NSLayoutAttributeCenterX, NSLayoutAttributeCenterY, NSLayoutAttributeWidth,
    NSLayoutAttributeHeight, NSLayoutAttributeTop, NSLayoutAttributeLeading,
    NSLayoutAttributeTrailing, NSLayoutAttributeBottom, NSLayoutConstraint,
    NSApplicationActivationPolicyRegular, NSApplicationActivationPolicyAccessory, NSObject, NSFont, NSBezelStyleRounded,
    NSTextFieldSquareBezel, NSFocusRingTypeNone, NSBox, NSBoxCustom, NSBoxSeparator,
    NSCursor, NSTrackingArea, NSTrackingMouseEnteredAndExited, NSTrackingActiveInActiveApp,
    NSWindowStyleMaskFullSizeContentView, NSWindowTitleHidden, NSButtonTypeMomentaryPushIn,
    NSWorkspace, NSImageView, NSImage
)
from Foundation import NSMakeRect, NSTimer, NSURL, NSURLRequest
from WebKit import WKWebView

from app import DB_PATH, BrainIndexer, WATCH_DIR, DownloadWatcherHandler
from watchdog.observers import Observer
from brain_search import perform_search, get_recent_files
try:
    # Optional: connect OCR utilities for future UI hooks
    from brain_ocr import ocr_image
except Exception:
    ocr_image = None

# --- UI Constants ---
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 500
ROW_HEIGHT = 50

class FlippedView(NSView):
    """A view with a top-left origin so scrolling to (0,0) goes to the top."""
    def isFlipped(self):
        return True

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
        self.model_lock = threading.Lock()
        self.search_timer = None
        self.is_indexing = False
        self.selected_path = None
        
        # Start filesystem watcher and initial sync (only add/remove)
        self.start_watchdog()
        
        # No bulk reindex on startup; watcher does a lightweight sync
        
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
        
        # Load Model (CPU) in background
        threading.Thread(target=self._get_shared_model, daemon=True).start()
        
        # Show recent files by default
        self.performSearch_("")

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
        
        # --- Split Pane ---
        self.splitView = NSSplitView.alloc().init()
        self.splitView.setTranslatesAutoresizingMaskIntoConstraints_(False)
        self.splitView.setDividerStyle_(1)  # thin divider
        self.splitView.setVertical_(True)   # left-right split
        self.visualEffectView.addSubview_(self.splitView)

        # Left Pane: Results list
        self.leftPane = NSView.alloc().init()
        self.leftPane.setTranslatesAutoresizingMaskIntoConstraints_(False)
        self.splitView.addSubview_(self.leftPane)

        self.scrollView = NSScrollView.alloc().init()
        self.scrollView.setTranslatesAutoresizingMaskIntoConstraints_(False)
        self.scrollView.setDrawsBackground_(False)
        self.scrollView.setHasVerticalScroller_(True)

        self.stackView = NSStackView.alloc().init()
        self.stackView.setOrientation_(1)
        self.stackView.setAlignment_(NSLayoutAttributeLeading)
        self.stackView.setSpacing_(5)
        self.stackView.setTranslatesAutoresizingMaskIntoConstraints_(False)

        self.documentView = FlippedView.alloc().init()
        self.documentView.setTranslatesAutoresizingMaskIntoConstraints_(False)
        self.documentView.addSubview_(self.stackView)

        self.scrollView.setDocumentView_(self.documentView)
        self.leftPane.addSubview_(self.scrollView)
        # Ensure initial scroll position is at the top
        try:
            cv = self.scrollView.contentView()
            cv.scrollToPoint_(NSPoint(0, 0))
            self.scrollView.reflectScrolledClipView_(cv)
        except Exception:
            pass

        # Right Pane: Preview + Metadata
        self.rightPane = NSView.alloc().init()
        self.rightPane.setTranslatesAutoresizingMaskIntoConstraints_(False)
        self.splitView.addSubview_(self.rightPane)

        # Preview image (for images/PDF first page via WebView separately)
        self.previewImageView = NSImageView.alloc().init()
        self.previewImageView.setTranslatesAutoresizingMaskIntoConstraints_(False)
        self.previewImageView.setImageScaling_(3)  # scale proportionally down
        self.rightPane.addSubview_(self.previewImageView)

        # Web view for PDF/HTML preview
        self.previewWebView = WKWebView.alloc().init()
        self.previewWebView.setTranslatesAutoresizingMaskIntoConstraints_(False)
        self.previewWebView.setValue_forKey_(False, "drawsBackground")
        self.rightPane.addSubview_(self.previewWebView)

        # Metadata labels
        self.metaTitle = NSTextField.labelWithString_("Metadata")
        self.metaTitle.setFont_(NSFont.systemFontOfSize_(15))
        self.metaTitle.setTextColor_(NSColor.whiteColor())
        self.metaTitle.setTranslatesAutoresizingMaskIntoConstraints_(False)
        self.rightPane.addSubview_(self.metaTitle)

        def makeLabelPair(name):
            key = NSTextField.labelWithString_(name)
            key.setTextColor_(NSColor.lightGrayColor())
            key.setTranslatesAutoresizingMaskIntoConstraints_(False)
            val = NSTextField.labelWithString_("")
            val.setTextColor_(NSColor.whiteColor())
            val.setTranslatesAutoresizingMaskIntoConstraints_(False)
            self.rightPane.addSubview_(key)
            self.rightPane.addSubview_(val)
            return key, val

        self.metaNameKey, self.metaNameVal = makeLabelPair("Name")
        self.metaWhereKey, self.metaWhereVal = makeLabelPair("Where")
        self.metaTypeKey, self.metaTypeVal = makeLabelPair("Type")
        self.metaSizeKey, self.metaSizeVal = makeLabelPair("Size")
        self.metaCreatedKey, self.metaCreatedVal = makeLabelPair("Created")
        self.metaModifiedKey, self.metaModifiedVal = makeLabelPair("Modified")

        # --- Analysis View (separate container, toggled by Nav) ---
        self.analysisContainer = NSView.alloc().init()
        self.analysisContainer.setTranslatesAutoresizingMaskIntoConstraints_(False)
        self.analysisContainer.setHidden_(True)
        self.visualEffectView.addSubview_(self.analysisContainer)

        self.webView = WKWebView.alloc().init()
        self.webView.setTranslatesAutoresizingMaskIntoConstraints_(False)
        self.webView.setValue_forKey_(False, "drawsBackground")
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
            
            # Split View fills below separator
            self.splitView.topAnchor().constraintEqualToAnchor_constant_(self.separator.bottomAnchor(), 0),
            self.splitView.leadingAnchor().constraintEqualToAnchor_(self.visualEffectView.leadingAnchor()),
            self.splitView.trailingAnchor().constraintEqualToAnchor_(self.visualEffectView.trailingAnchor()),
            self.splitView.bottomAnchor().constraintEqualToAnchor_(self.visualEffectView.bottomAnchor()),

            # Left Pane
            self.scrollView.topAnchor().constraintEqualToAnchor_constant_(self.leftPane.topAnchor(), 0),
            self.scrollView.leadingAnchor().constraintEqualToAnchor_(self.leftPane.leadingAnchor()),
            self.scrollView.trailingAnchor().constraintEqualToAnchor_(self.leftPane.trailingAnchor()),
            self.scrollView.bottomAnchor().constraintEqualToAnchor_(self.leftPane.bottomAnchor()),

            self.stackView.topAnchor().constraintEqualToAnchor_(self.documentView.topAnchor()),
            self.stackView.leadingAnchor().constraintEqualToAnchor_(self.documentView.leadingAnchor()),
            self.stackView.trailingAnchor().constraintEqualToAnchor_(self.documentView.trailingAnchor()),
            self.documentView.widthAnchor().constraintEqualToAnchor_(self.scrollView.widthAnchor()),
            self.documentView.heightAnchor().constraintGreaterThanOrEqualToAnchor_(self.scrollView.heightAnchor()),

            # Right Pane layout
            self.previewImageView.topAnchor().constraintEqualToAnchor_constant_(self.rightPane.topAnchor(), 10),
            self.previewImageView.centerXAnchor().constraintEqualToAnchor_(self.rightPane.centerXAnchor()),
            self.previewImageView.heightAnchor().constraintEqualToConstant_(260),
            self.previewImageView.widthAnchor().constraintEqualToConstant_(360),

            self.previewWebView.topAnchor().constraintEqualToAnchor_constant_(self.rightPane.topAnchor(), 10),
            self.previewWebView.leadingAnchor().constraintEqualToAnchor_(self.rightPane.leadingAnchor()),
            self.previewWebView.trailingAnchor().constraintEqualToAnchor_(self.rightPane.trailingAnchor()),
            self.previewWebView.heightAnchor().constraintEqualToConstant_(260),

            self.metaTitle.topAnchor().constraintEqualToAnchor_constant_(self.previewWebView.bottomAnchor(), 20),
            self.metaTitle.leadingAnchor().constraintEqualToAnchor_constant_(self.rightPane.leadingAnchor(), 20),

            self.metaNameKey.topAnchor().constraintEqualToAnchor_constant_(self.metaTitle.bottomAnchor(), 12),
            self.metaNameKey.leadingAnchor().constraintEqualToAnchor_constant_(self.rightPane.leadingAnchor(), 20),
            self.metaNameVal.centerYAnchor().constraintEqualToAnchor_(self.metaNameKey.centerYAnchor()),
            self.metaNameVal.leadingAnchor().constraintEqualToAnchor_constant_(self.metaNameKey.trailingAnchor(), 20),

            self.metaWhereKey.topAnchor().constraintEqualToAnchor_constant_(self.metaNameKey.bottomAnchor(), 8),
            self.metaWhereKey.leadingAnchor().constraintEqualToAnchor_constant_(self.rightPane.leadingAnchor(), 20),
            self.metaWhereVal.centerYAnchor().constraintEqualToAnchor_(self.metaWhereKey.centerYAnchor()),
            self.metaWhereVal.leadingAnchor().constraintEqualToAnchor_constant_(self.metaWhereKey.trailingAnchor(), 20),

            self.metaTypeKey.topAnchor().constraintEqualToAnchor_constant_(self.metaWhereKey.bottomAnchor(), 8),
            self.metaTypeKey.leadingAnchor().constraintEqualToAnchor_constant_(self.rightPane.leadingAnchor(), 20),
            self.metaTypeVal.centerYAnchor().constraintEqualToAnchor_(self.metaTypeKey.centerYAnchor()),
            self.metaTypeVal.leadingAnchor().constraintEqualToAnchor_constant_(self.metaTypeKey.trailingAnchor(), 20),

            self.metaSizeKey.topAnchor().constraintEqualToAnchor_constant_(self.metaTypeKey.bottomAnchor(), 8),
            self.metaSizeKey.leadingAnchor().constraintEqualToAnchor_constant_(self.rightPane.leadingAnchor(), 20),
            self.metaSizeVal.centerYAnchor().constraintEqualToAnchor_(self.metaSizeKey.centerYAnchor()),
            self.metaSizeVal.leadingAnchor().constraintEqualToAnchor_constant_(self.metaSizeKey.trailingAnchor(), 20),

            self.metaCreatedKey.topAnchor().constraintEqualToAnchor_constant_(self.metaSizeKey.bottomAnchor(), 8),
            self.metaCreatedKey.leadingAnchor().constraintEqualToAnchor_constant_(self.rightPane.leadingAnchor(), 20),
            self.metaCreatedVal.centerYAnchor().constraintEqualToAnchor_(self.metaCreatedKey.centerYAnchor()),
            self.metaCreatedVal.leadingAnchor().constraintEqualToAnchor_constant_(self.metaCreatedKey.trailingAnchor(), 20),

            self.metaModifiedKey.topAnchor().constraintEqualToAnchor_constant_(self.metaCreatedKey.bottomAnchor(), 8),
            self.metaModifiedKey.leadingAnchor().constraintEqualToAnchor_constant_(self.rightPane.leadingAnchor(), 20),
            self.metaModifiedVal.centerYAnchor().constraintEqualToAnchor_(self.metaModifiedKey.centerYAnchor()),
            self.metaModifiedVal.leadingAnchor().constraintEqualToAnchor_constant_(self.metaModifiedKey.trailingAnchor(), 20)
            ,
            # Analysis Container fills area (hidden unless Analysis is selected)
            self.analysisContainer.topAnchor().constraintEqualToAnchor_constant_(self.navStackView.bottomAnchor(), 20),
            self.analysisContainer.leadingAnchor().constraintEqualToAnchor_(self.visualEffectView.leadingAnchor()),
            self.analysisContainer.trailingAnchor().constraintEqualToAnchor_(self.visualEffectView.trailingAnchor()),
            self.analysisContainer.bottomAnchor().constraintEqualToAnchor_(self.visualEffectView.bottomAnchor()),

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
        self.splitView.setHidden_(not visible)

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
            self.performSelectorOnMainThread_withObject_waitUntilDone_("updateStatus:", "Cleaning duplicates...", False)
            indexer = BrainIndexer(DB_PATH, model=self.model)
            # One-time cleanup to remove duplicates
            try:
                indexer.dedupe_index()
            except Exception as e:
                print(f"Cleanup error: {e}")
            
            self.performSelectorOnMainThread_withObject_waitUntilDone_("updateStatus:", "Checking files...", False)
            indexer.sync_index(Path(WATCH_DIR))  # Only add new or remove missing
            indexer.close()
            self.performSelectorOnMainThread_withObject_waitUntilDone_("updateStatus:", "Search your brain...", False)
            # Refresh current view (keep search text)
            self.performSelectorOnMainThread_withObject_waitUntilDone_("refreshView:", None, False)
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
        # Empty query → show recent files
        if not query or len(query) < 2:
            threading.Thread(target=self._load_recent_files_thread, daemon=True).start()
            return
        
        # Ensure model is available
        if not self.model:
            model = self._get_shared_model()
            if not model:
                print("Model loading; search deferred")
                return
        
        threading.Thread(target=self._search_thread, args=(query,), daemon=True).start()

    @objc.python_method
    def _search_thread(self, query):
        try:
            results = perform_search(query, self.model, DB_PATH)
            self.performSelectorOnMainThread_withObject_waitUntilDone_("updateResults:", results, False)
            
        except Exception as e:
            print(f"Search error: {e}")

    @objc.python_method
    def _load_recent_files_thread(self):
        try:
            results = get_recent_files(DB_PATH)
            self.performSelectorOnMainThread_withObject_waitUntilDone_("updateResults:", results, False)
        except Exception as e:
            print(f"Recent files error: {e}")

    def updateResults_(self, results):
        self.clearResults()
        self.selected_path = None
        
        for i, res in enumerate(results):
            # Use click-to-select/open callback
            row = SearchResultRow.alloc().initWithResult_callback_(res, self.onResultClicked_)
            self.stackView.addArrangedSubview_(row)
            row.setWantsLayer_(True)
            
            # Width constraint
            row.widthAnchor().constraintEqualToAnchor_(self.stackView.widthAnchor()).setActive_(True)
            
            # Auto-select first result
            if i == 0:
                self.onResultClicked_(res[2])
        # Ensure scroll starts at the top
        try:
            content_view = self.scrollView.contentView()
            content_view.scrollToPoint_(NSPoint(0, 0))
            self.scrollView.reflectScrolledClipView_(content_view)
        except Exception:
            pass

    @objc.python_method
    def clearResults(self):
        for view in self.stackView.arrangedSubviews():
            self.stackView.removeView_(view)
            view.removeFromSuperview()

    @objc.python_method
    def onResultClicked_(self, path):
        # First click selects; clicking selected opens
        if self.selected_path == path:
            try:
                NSWorkspace.sharedWorkspace().openFile_(path)
            except Exception as e:
                print(f"Open error: {e}")
            return
        
        self.selected_path = path
        self.updateSelectionVisuals()
        self.updatePreview_(path)
    
    @objc.python_method
    def updateSelectionVisuals(self):
        for view in self.stackView.arrangedSubviews():
            # Best-effort style highlight
            try:
                if hasattr(view, 'path') and view.path == self.selected_path:
                    view.layer().setBackgroundColor_(NSColor.selectedControlColor().CGColor())
                else:
                    view.layer().setBackgroundColor_(NSColor.clearColor().CGColor())
            except Exception:
                pass

    @objc.python_method
    def updatePreview_(self, path):
        try:
            p = Path(path)
            # Metadata
            self.metaNameVal.setStringValue_(p.name)
            self.metaWhereVal.setStringValue_(str(p.parent))
            self.metaTypeVal.setStringValue_(p.suffix.lower() or "")
            try:
                sz = p.stat().st_size
                self.metaSizeVal.setStringValue_(f"{sz} bytes")
                try:
                    created = datetime.fromtimestamp(p.stat().st_birthtime)
                except AttributeError:
                    created = datetime.fromtimestamp(p.stat().st_ctime)
                modified = datetime.fromtimestamp(p.stat().st_mtime)
                self.metaCreatedVal.setStringValue_(created.strftime("%b %d, %Y at %I:%M %p"))
                self.metaModifiedVal.setStringValue_(modified.strftime("%b %d, %Y at %I:%M %p"))
            except Exception:
                pass

            # Reset preview widgets visibility
            self.previewImageView.setHidden_(True)
            self.previewWebView.setHidden_(True)

            ext = p.suffix.lower()
            if ext in {".png", ".jpg", ".jpeg"}:
                try:
                    img = NSImage.alloc().initWithContentsOfFile_(str(p))
                    if img:
                        self.previewImageView.setImage_(img)
                        self.previewImageView.setHidden_(False)
                except Exception:
                    pass
            elif ext in {".pdf", ".html", ".htm"}:
                try:
                    url = NSURL.fileURLWithPath_(str(p))
                    req = NSURLRequest.requestWithURL_(url)
                    self.previewWebView.loadRequest_(req)
                    self.previewWebView.setHidden_(False)
                except Exception:
                    pass
            else:
                # For text-like files, show small HTML preview via WebView using snippet from DB
                try:
                    conn = duckdb.connect(str(DB_PATH))
                    row = conn.execute("SELECT text_snippet FROM files_index WHERE path = ?", [str(p.absolute())]).fetchone()
                    conn.close()
                    snippet = row[0] if row and row[0] else ""
                    html = f"<html><body style='background:#1e1e1e;color:#ddd;font-family:system-ui;padding:12px;'><pre style='white-space:pre-wrap'>{snippet}</pre></body></html>"
                    self.previewWebView.loadHTMLString_baseURL_(html, None)
                    self.previewWebView.setHidden_(False)
                except Exception:
                    pass
        except Exception as e:
            print(f"Preview error: {e}")

    @objc.python_method
    def start_watchdog(self):
        try:
            # Initialize indexer and sync on startup (no full reindex)
            self.indexer = BrainIndexer(DB_PATH, model=self.model)
            self.indexer.sync_index(Path(WATCH_DIR))
            
            # Watch for changes and refresh view when files are added/removed
            self.observer = Observer()
            handler = DownloadWatcherHandler(self.indexer, callback=self.onFileChanged)
            self.observer.schedule(handler, str(WATCH_DIR), recursive=False)
            self.observer.start()
            print(f"Watching {WATCH_DIR} for changes.")
        except Exception as e:
            print(f"Error starting watchdog: {e}")

    @objc.python_method
    def onFileChanged(self):
        # Called on add/delete → refresh current view without losing context
        self.performSelectorOnMainThread_withObject_waitUntilDone_("refreshView:", None, False)

    def refreshView_(self, sender):
        query = self.searchField.stringValue()
        self.performSearch_(query)

    def ocrImageAtPath_(self, path):
        """Run OCR on an image path and print a short preview."""
        try:
            if ocr_image is None:
                print("OCR not available. Install pyobjc-framework-Vision or pytesseract+Pillow.")
                return
            text = ocr_image(Path(path))
            if text:
                preview = (text[:200] + "...") if len(text) > 200 else text
                print(f"OCR Preview for {path}:\n{preview}")
            else:
                print("No text detected.")
        except Exception as e:
            print(f"OCR error: {e}")

    def applicationWillTerminate_(self, notification):
        if hasattr(self, 'streamlit_process') and self.streamlit_process:
            self.streamlit_process.terminate()

    @objc.python_method
    def _get_shared_model(self):
        with self.model_lock:
            if self.model is None:
                try:
                    # CPU to reduce memory
                    self.model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
                    print("Model loaded (CPU)")
                except Exception as e:
                    print(f"Error loading model: {e}")
            return self.model

if __name__ == "__main__":
    app = NSApplication.sharedApplication()
    delegate = BrainApp.alloc().init()
    app.setDelegate_(delegate)
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    app.activateIgnoringOtherApps_(True)
    app.run()
