# Vercel Deployment Setup - Verified Configuration

## File Structure

```
atharva-portfolio/
├── index.py          # Root entry point (Vercel auto-detects FastAPI here)
├── app.py            # Main FastAPI application
├── vercel.json       # Vercel configuration
├── requirements.txt  # Python dependencies
├── static/
│   └── styles.css   # Static CSS files
├── templates/
│   └── index.html   # Jinja2 templates
├── api/
│   └── index.py     # Backup entry point (not used in current config)
└── test_local.py    # Local testing script
```

## Configuration Details

### 1. Root Entry Point (`index.py`)

- **Purpose**: Vercel automatically detects FastAPI apps at recognized entrypoints
- **Content**: Simply imports `app` from `app.py`
- **Why**: Vercel's Python runtime has built-in ASGI support and auto-detects FastAPI

### 2. Vercel Configuration (`vercel.json`)

```json
{
  "version": 2,
  "builds": [
    {
      "src": "index.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/static/(.*)",
      "dest": "/static/$1"
    },
    {
      "src": "/(.*)",
      "dest": "/index.py"
    }
  ]
}
```

**Key Points**:

- Builds `index.py` using `@vercel/python` runtime
- Routes `/static/*` to static files (served directly by Vercel)
- Routes all other requests to `/index.py` (FastAPI handler)

### 3. FastAPI App (`app.py`)

- Uses `Path(__file__).parent` for path resolution (works in both local and Vercel)
- Mounts static files if directory exists
- Uses Jinja2Templates for HTML rendering

## Testing Locally

### Quick Test

```bash
python test_local.py
```

This verifies:

- ✓ Imports work correctly
- ✓ Paths resolve properly
- ✓ Routes are configured
- ✓ Static/template directories exist

### Standard FastAPI (Development)

```bash
uvicorn app:app --reload
```

Visit: `http://127.0.0.1:8000/`

### Vercel Dev (Production-like)

```bash
vercel dev
```

Follow prompts and visit the provided URL (usually `http://localhost:3000`)

## Deployment Checklist

Before deploying to Vercel:

- [x] `index.py` exists at root and imports `app` correctly
- [x] `vercel.json` configured correctly
- [x] `requirements.txt` includes all dependencies
- [x] Static files in `static/` directory
- [x] Templates in `templates/` directory
- [x] `python test_local.py` passes
- [x] `uvicorn app:app --reload` works locally

## Common Issues & Solutions

### Issue: `TypeError: issubclass() arg 1 must be a class`

**Solution**: Use root-level `index.py` instead of `api/index.py`. Vercel auto-detects FastAPI at recognized entrypoints.

### Issue: Static files not loading

**Solution**: Ensure `vercel.json` has the `/static/*` route configured, and files are in the `static/` directory.

### Issue: Template not found

**Solution**: Verify `TEMPLATES_DIR` path resolves correctly. Use `Path(__file__).parent` for reliable path resolution.

## Why This Setup Works

1. **Vercel Auto-Detection**: Vercel's Python runtime automatically detects FastAPI apps at recognized entrypoints (`index.py`, `app.py`, etc.)

2. **No Mangum Needed**: Vercel has built-in ASGI support, so we don't need Mangum adapter

3. **Path Resolution**: Using `Path(__file__).parent` ensures paths work in both local development and Vercel's serverless environment

4. **Static Files**: Vercel serves `/static/*` directly for better performance, while FastAPI can also serve them as fallback
