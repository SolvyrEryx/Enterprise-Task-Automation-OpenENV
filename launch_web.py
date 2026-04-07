#!/usr/bin/env python
"""Simple web interface launcher"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app import create_demo_interface

if __name__ == "__main__":
    print("=" * 70)
    print("ENTERPRISE TASK AUTOMATION - WEB INTERFACE")
    print("=" * 70)
    print("\nStarting Gradio web interface...")
    print("Open your browser to: http://localhost:7860\n")
    
    try:
        demo = create_demo_interface()
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True,
        )
    except Exception as e:
        print(f"\nError starting server: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure port 7860 is not in use")
        print("2. Check that Gradio is installed: pip install gradio")
        print("3. Try a different port: change server_port above")
        sys.exit(1)
