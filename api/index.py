"""
RakaRAM - Vercel Serverless Function Entrypoint
===============================================
Exports the FastAPI app instance for Vercel's standard API runtime convention (/api/index.py).
"""
import sys
from pathlib import Path

# Ensure project root is in sys.path so src, data, and saved_models are discoverable
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from main import app, application, handler
