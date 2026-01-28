#!/usr/bin/env python3
"""
Build a standalone static HTML file from the portfolio dashboard.
Run: python build_static.py
Output: portfolio.html (standalone file with embedded CSS)
"""

import sys
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

# Import the computation logic from app.py
from app import compute_portfolio_snapshot, APP_TITLE

def build_static_html(output_path: str = "portfolio.html"):
    """Generate a standalone HTML file with embedded CSS."""
    
    print("Fetching portfolio data...")
    try:
        snapshot = compute_portfolio_snapshot()
    except Exception as e:
        print(f"Error computing portfolio snapshot: {e}", file=sys.stderr)
        sys.exit(1)
    
    print("Loading template...")
    template_dir = Path("templates")
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    template = env.get_template("index.html")
    
    print("Reading CSS...")
    css_path = Path("static/styles.css")
    if not css_path.exists():
        print(f"Error: CSS file not found at {css_path}", file=sys.stderr)
        sys.exit(1)
    
    with open(css_path, "r", encoding="utf-8") as f:
        css_content = f.read()
    
    print("Rendering HTML...")
    # Render the template
    html_content = template.render(
        title=APP_TITLE,
        snapshot=snapshot,
    )
    
    # Replace the CSS link with embedded CSS
    html_content = html_content.replace(
        '<link rel="stylesheet" href="/static/styles.css" />',
        f'<style>\n{css_content}\n</style>'
    )
    
    print(f"Writing to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"✓ Successfully generated {output_path}")
    print(f"  Open it in your browser or deploy to any static hosting service.")


if __name__ == "__main__":
    output_file = sys.argv[1] if len(sys.argv) > 1 else "portfolio.html"
    build_static_html(output_file)
