from http.server import BaseHTTPRequestHandler


class handler(BaseHTTPRequestHandler):
    """
    Minimal Vercel Python function.

    NOTE:
    - Vercel's serverless runtime is not designed for long‑running Streamlit
      servers, so this endpoint only returns a helpful message explaining
      that the app itself should be run with `streamlit run app.py`.
    - For a fully supported Streamlit deployment, prefer Streamlit
      Community Cloud or another full‑server host.
    """

    def do_GET(self):
        body = (
            "This project is a Streamlit app.\n\n"
            "To run locally:\n"
            "  streamlit run app.py\n\n"
            "For production hosting, use Streamlit Community Cloud or a full "
            "Python host. Vercel serverless functions cannot run a persistent "
            "Streamlit server reliably."
        )

        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body.encode("utf-8"))))
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

