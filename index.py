"""
Root-level entry point for Vercel.
Vercel automatically detects FastAPI apps at recognized entrypoints like index.py.
"""
from app import app

# Vercel will automatically detect this FastAPI app instance
