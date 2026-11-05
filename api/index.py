"""
RakaRAM - Vercel Serverless Function Entrypoint
===============================================
Exports the FastAPI app instance for Vercel's standard API runtime convention (/api/index.py).
Includes cold-start error protection to prevent opaque 500 errors.
"""
import sys
import traceback
from pathlib import Path

# Ensure project root is in sys.path so modules can be imported
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from main import app, application
except Exception:
    err_tb = traceback.format_exc()
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse
    
    app = FastAPI(title="RakaRAM Cold Start Recovery")
    application = app

    @app.api_route("/{path:path}", methods=["GET", "POST", "HEAD"])
    def recovery_handler(path: str = ""):
        return HTMLResponse(
            f"""<!DOCTYPE html>
            <html lang="en">
            <head>
              <meta name="viewport" content="width=device-width, initial-scale=1.0">
              <title>RakaRAM - Service Initialization Notice</title>
              <style>
                body {{ background: #090D16; color: #F8FAFC; font-family: -apple-system, sans-serif; padding: 24px; line-height: 1.6; }}
                .box {{ max-width: 680px; margin: 20px auto; background: #111726; border: 1px solid #1E293B; border-radius: 12px; padding: 20px; }}
                h2 {{ color: #10B981; margin-bottom: 8px; }}
                pre {{ background: #000; padding: 14px; border-radius: 8px; color: #F87171; overflow-x: auto; font-size: 13px; }}
              </style>
            </head>
            <body>
              <div class="box">
                <h2>⚡ RakaRAM Service Starting Up</h2>
                <p>The service encountered an initialization issue during container cold start:</p>
                <pre>{err_tb}</pre>
              </div>
            </body>
            </html>""",
            status_code=200
        )
