My Portfolio – FastAPI Dashboard

This project exposes a read‑only, static HTML dashboard for Atharva's portfolio using FastAPI, Jinja2 templates, and plain HTML/CSS. Data is sourced from a published Google Sheet and Yahoo Finance.

## Running locally

1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the FastAPI app:

```bash
uvicorn app:app --reload
```

4. Open the dashboard in your browser at:

```text
http://127.0.0.1:8000/
```

The output is a static server‑rendered HTML page (no client‑side JavaScript), recomputed on each request using the latest available data.

## Building a standalone static HTML file

To generate a single standalone HTML file that can be opened directly in a browser or deployed to any static hosting service:

```bash
python build_static.py
```

This will create `portfolio.html` in the current directory with all CSS embedded inline. You can:

- Open `portfolio.html` directly in your browser
- Deploy it to GitHub Pages, Netlify, Vercel, or any static hosting service
- Share it as a single file

The static file contains a snapshot of the portfolio data at the time it was generated. To update it, simply run the build script again.
