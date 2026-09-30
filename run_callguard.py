"""
CallGuard NPU Root Convenience Runner.

Design:
    Enables quick execution from the project root directory via `python run_callguard.py`.
"""

import sys
import os

# Ensure current directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from callguard.main import main

if __name__ == "__main__":
    sys.exit(main())
