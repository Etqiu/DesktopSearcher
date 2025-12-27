import objc
from Cocoa import NSApplication
from WebKit import WKWebView
import sys

try:
    app = NSApplication.sharedApplication()
    wv = WKWebView.alloc().init()
    print("WKWebView created successfully")
except Exception as e:
    print(f"Error: {e}")
