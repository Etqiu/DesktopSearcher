import pytest
from pathlib import Path
import sys
import os

# Add src to path so we can import brain_ocr
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from brain_ocr import ocr_image

def test_ocr_on_downloads_pngs():
    downloads_dir = Path.home() / "Downloads"
    png_files = list(downloads_dir.glob("*.png"))
    
    if not png_files:
        pytest.skip("No PNG files found in Downloads folder")
        
    print(f"Found {len(png_files)} PNG files in {downloads_dir}")
    
    for png_file in png_files:
        print(f"Testing OCR on: {png_file}")
        text = ocr_image(png_file)
        
        # We just want to verify it runs without error and returns a string (empty or not)
        # Ideally it should return some text if the image has text, but for now we check type
        assert text is not None, f"OCR failed (returned None) for {png_file}"
        assert isinstance(text, str), f"OCR returned non-string for {png_file}"
        
        # Optional: print a snippet of the text for manual verification in logs
        snippet = text[:100].replace('\n', ' ')
        print(f"  Result snippet: {snippet}...")

if __name__ == "__main__":
    # Allow running directly
    test_ocr_on_downloads_pngs()
