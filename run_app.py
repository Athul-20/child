"""
Runner script for the Malayalam Child Handwriting Diagnostic & Longitudinal Tracking Engine.
Ensures UTF-8 encoding for Malayalam Unicode glyphs on Windows and boots the FastAPI server.
"""

import os
import sys
import time
import webbrowser
import threading

# Force UTF-8 encoding across Windows console and I/O streams
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

def open_browser_delayed(url: str, delay_seconds: float = 2.5):
    """Waits for the server to bind before opening the browser."""
    time.sleep(delay_seconds)
    try:
        webbrowser.open(url)
    except Exception:
        pass

if __name__ == "__main__":
    import uvicorn
    
    host = "127.0.0.1"
    port = 8000
    url = f"http://{host}:{port}"
    
    print("=" * 80)
    print("MALAYALAM AGENTIC AI DIAGNOSTIC ENGINE")
    print("Multi-Modal Handwriting Perception | 3-Tier G2P & PanPhon | CMSDA Closed-Loop")
    print("=" * 80)
    print(f"Starting server at: {url}")
    print("Ground truth dataset: Malayalam_Data_With_Generated_IPA_ErrorsHandled_updated (1).xlsx (8,673 rows)")
    print("Opening browser automatically...")
    print("=" * 80)
    
    # Launch browser thread
    threading.Thread(target=open_browser_delayed, args=(url, 3.0), daemon=True).start()
    
    # Run uvicorn server
    uvicorn.run(
        "app.server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )

