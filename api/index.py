"""
Vercel serverless function entry point for FastAPI app.
"""
import sys
import os
from pathlib import Path

# Add parent directory to path so we can import app
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Change to parent directory to ensure relative paths work
os.chdir(str(parent_dir))

from app import app

# For Vercel: export the app variable at module level
# Vercel's Python runtime will detect FastAPI and handle it as ASGI
