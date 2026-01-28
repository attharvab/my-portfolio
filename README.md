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

## Deploying to Vercel

This FastAPI app is configured to deploy on Vercel as a serverless function.

### Prerequisites

1. Install Vercel CLI (optional, for local testing):

   ```bash
   npm i -g vercel
   ```

2. Make sure your project is in a git repository (Vercel works best with git).

### Deployment Steps

1. **Deploy via Vercel Dashboard** (recommended):
   - Go to [vercel.com](https://vercel.com)
   - Click "New Project"
   - Import your git repository
   - Vercel will auto-detect the Python/FastAPI setup from `vercel.json`
   - Click "Deploy"

2. **Deploy via CLI**:

   ```bash
   vercel
   ```

   Follow the prompts to link your project.

3. **Production deployment**:
   ```bash
   vercel --prod
   ```

### How It Works

- The `index.py` file at the root imports your FastAPI app
- Vercel automatically detects FastAPI apps at recognized entrypoints (`index.py`, `app.py`, etc.)
- `vercel.json` configures routes and builds
- Static files (`/static/*`) are served directly by Vercel for better performance
- All other routes are handled by the FastAPI serverless function

### Environment Variables

If you need to change the Google Sheet URL or other configuration, you can set environment variables in the Vercel dashboard under Project Settings → Environment Variables.

### Local Testing

Test the Vercel setup locally:

```bash
vercel dev
```

This will start a local server that mimics Vercel's serverless environment.
