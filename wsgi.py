"""
PythonAnywhere WSGI entrypoint.

In the Web tab → WSGI configuration file, replace the default contents with:

    import sys
    path = '/home/YOUR_USERNAME/cloud'   # or .../ai_management/cloud
    if path not in sys.path:
        sys.path.insert(0, path)
    from wsgi import application
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

CLOUD_DIR = Path(__file__).resolve().parent
if str(CLOUD_DIR) not in sys.path:
    sys.path.insert(0, str(CLOUD_DIR))

_env = CLOUD_DIR / '.env'
if _env.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env)
    except ImportError:
        for line in _env.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            k, _, v = line.partition('=')
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

os.environ.setdefault('EDGE_SYNC_SECRET', 'change-me-edge-sync')
os.environ.setdefault('JWT_SECRET', 'change-me-jwt-on-pa')
os.environ.setdefault('FRONTEND_DIST', str(CLOUD_DIR / 'static'))

from app import app as application  # noqa: E402
