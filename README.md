# my-portfolio

## Local development

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploying to Vercel (experimental)

This repository includes an **experimental** Vercel setup for the app:

- `vercel.json` routes all traffic to `api/index.py`.
- `api/index.py` is a minimal Python function so Vercel can build the project.

⚠️ **Important:** Vercel’s serverless Python runtime is not designed to host a long‑running Streamlit server, so this setup only proves that the project can build on Vercel. It does **not** run the interactive Streamlit UI in production.

### Steps

1. Push this repo to GitHub.
2. In the Vercel dashboard:
   - “Add New Project” → import the GitHub repo.
   - Use the default settings (Vercel will detect `vercel.json`).
3. Vercel will deploy an endpoint at your project URL that returns a short message explaining that the app is a Streamlit project.

For a **fully working hosted version of the Streamlit UI**, use **Streamlit Community Cloud** or another full Python host instead of Vercel.
