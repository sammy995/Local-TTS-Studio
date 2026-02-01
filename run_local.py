"""
Local TTS Studio - Startup Script
Launches the refactored local runtime
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from runtimes.local_api import start_server

if __name__ == "__main__":
    print("=" * 60)
    print("Local TTS Studio - Starting...")
    print("=" * 60)
    start_server()
