#!/usr/bin/env python
"""Simple web interface launcher"""

import sys
from pathlib import Path
import uvicorn

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import the FastAPI 'app' instead of the Gradio 'demo'
from app import app

if __name__ == "__main__":
    print("=" * 70)
    print("ENTERPRISE TASK AUTOMATION - WEB & API SERVER")
    print("=" * 70)
    print("\nStarting Uvicorn server...")
    print("Open your browser to: http://localhost:7860 (UI)")
    print("API endpoints available at: http://localhost:7860/docs\n")
    
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=7860,
        )
    except Exception as e:
        print(f"\nError starting server: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure port 7860 is not in use")
        print("2. Check that FastAPI and Uvicorn are installed: pip install fastapi uvicorn")
        sys.exit(1)
