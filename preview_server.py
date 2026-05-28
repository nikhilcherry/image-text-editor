"""
Lightweight static preview server for the UI.
Serves index.html at / and static assets at /static/…
(No Flask deps needed — used during development/preview only.)
"""
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler

ROOT = os.path.join(os.path.dirname(__file__), "app")
os.chdir(ROOT)

class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "":
            self.path = "/templates/index.html"
        return super().do_GET()

    def log_message(self, *_):
        pass   # silence request log noise

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print(f"Preview server: http://localhost:{port}")
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()
