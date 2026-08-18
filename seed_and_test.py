"""
Seed sample warehouse data and verify cloud edge-sync + video relay.

Usage (from the cloud/ folder):

  python seed_and_test.py
  python seed_and_test.py --live http://127.0.0.1:5000

Exit code 0 = all checks passed.
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import sys
import time
from pathlib import Path

CLOUD_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CLOUD_DIR))

# Ensure secrets for local test
os.environ.setdefault('EDGE_SYNC_SECRET', 'change-me-edge-sync')
os.environ.setdefault('SEED_SAMPLE_DATA', '1')
os.environ.setdefault('JWT_SECRET', 'test-jwt')


def _make_jpeg(label: str = 'SAMPLE') -> bytes:
    from PIL import Image, ImageDraw
    img = Image.new('RGB', (640, 360), color=(30, 40, 55))
    draw = ImageDraw.Draw(img)
    draw.rectangle([40, 40, 600, 320], outline=(50, 200, 80), width=4)
    draw.text((60, 60), f'AI CCTV sample frame — {label}', fill=(240, 240, 240))
    draw.text((60, 100), time.strftime('%Y-%m-%d %H:%M:%S'), fill=(180, 220, 255))
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=70)
    return buf.getvalue()


class Client:
    def __init__(self, base: str | None):
        self.base = (base or '').rstrip('/')
        if self.base:
            import urllib.request
            self._urllib = urllib.request
            self.mode = 'live'
        else:
            from app import app, Config
            self.app = app
            self.Config = Config
            self.client = app.test_client()
            self.mode = 'test'

    @property
    def secret(self) -> str:
        if self.mode == 'test':
            return self.Config.EDGE_SYNC_SECRET
        return os.environ.get('EDGE_SYNC_SECRET', 'change-me-edge-sync')

    def request(self, method: str, path: str, **kwargs):
        if self.mode == 'test':
            fn = getattr(self.client, method.lower())
            headers = kwargs.pop('headers', {})
            json_body = kwargs.pop('json', None)
            data = kwargs.pop('data', None)
            return fn(path, headers=headers, json=json_body, data=data)
        import json as _json
        import urllib.error
        url = self.base + path
        headers = dict(kwargs.get('headers') or {})
        body = None
        if 'json' in kwargs and kwargs['json'] is not None:
            body = _json.dumps(kwargs['json']).encode()
            headers.setdefault('Content-Type', 'application/json')
        req = self._urllib.Request(url, data=body, headers=headers, method=method.upper())
        try:
            with self._urllib.urlopen(req, timeout=15) as resp:
                raw = resp.read()
                class R:
                    status_code = resp.status
                    data = raw
                    headers = {k.lower(): v for k, v in resp.headers.items()}
                    def get_json(self):
                        return _json.loads(raw.decode() or '{}')
                    @property
                    def json(self):
                        return self.get_json()
                    @property
                    def content_type(self):
                        return self.headers.get('content-type', '')
                return R()
        except urllib.error.HTTPError as e:
            class R:
                status_code = e.code
                data = e.read()
                def get_json(self):
                    try:
                        return _json.loads(self.data.decode() or '{}')
                    except Exception:
                        return {}
                @property
                def json(self):
                    return self.get_json()
                content_type = e.headers.get('content-type', '')
            return R()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--live', metavar='URL', help='Running cloud base URL')
    args = parser.parse_args()

    print('=' * 60)
    print(' AI CCTV cloud — seed + sample tests')
    print('=' * 60)

    c = Client(args.live)
    secret = c.secret
    fails = 0

    def check(name: str, ok: bool, detail: str = ''):
        nonlocal fails
        mark = 'PASS' if ok else 'FAIL'
        print(f'  [{mark}] {name}' + (f' — {detail}' if detail else ''))
        if not ok:
            fails += 1

    # 1) Health
    r = c.request('GET', '/api/health')
    check('health', r.status_code == 200 and (r.json or {}).get('status') == 'ok', str(r.json))

    # 2) Seed sample (via import when test mode)
    if c.mode == 'test':
        from core import db
        db.init_schema()
        created = db.seed_sample_demo()
        check('seed_sample_demo', True, str(created))
    else:
        check('seed_sample_demo', True, 'skipped on --live (set SEED_SAMPLE_DATA=1 on server)')

    # 3) Push sample frame (simulates Windows edge)
    jpeg = _make_jpeg('edge-sync')
    r = c.request(
        'POST', '/api/edge/frame',
        headers={'X-Edge-Secret': secret},
        json={'frame': base64.b64encode(jpeg).decode()},
    )
    check('edge/frame push', r.status_code == 200, str(getattr(r, 'json', None)))

    # 4) Snapshot
    r = c.request('GET', '/api/video_feed/snapshot')
    check('video_feed/snapshot', r.status_code == 200 and r.data[:2] == b'\xff\xd8', f'{len(r.data)} bytes')

    # 5) Sync detections + session
    r = c.request(
        'POST', '/api/edge/sync',
        headers={'X-Edge-Secret': secret, 'Content-Type': 'application/json'},
        json={
            'detections': [
                {'type': 'sugar_bag', 'direction': 'IN', 'confidence': 0.95, 'count': 2},
                {'type': 'sugar_bag', 'direction': 'OUT', 'confidence': 0.88, 'count': 1},
            ],
            'session': {'session_in': 12, 'session_out': 7, 'line_position': 0.45},
        },
    )
    check('edge/sync', r.status_code == 200 and (r.json or {}).get('inserted', 0) >= 2, str(r.json))

    # 6) Camera status (relay active)
    r = c.request('GET', '/api/camera/status')
    body = r.json or {}
    check('camera/status relay', r.status_code == 200 and body.get('mode') == 'cloud_relay' and body.get('active') is True,
          str(body))

    # 7) Stats / inventory / trucks
    for path in ('/api/stats', '/api/inventory', '/api/trucks', '/api/detections?limit=10'):
        r = c.request('GET', path)
        check(path, r.status_code == 200, f'status={r.status_code}')

    # 8) MJPEG header (test client only — live would hang on stream)
    if c.mode == 'test':
        with c.client.get('/api/video_feed') as resp:
            ctype = resp.content_type or ''
            check('video_feed MJPEG', 'multipart/x-mixed-replace' in ctype, ctype)
            chunk = next(resp.response)
            check('video_feed first JPEG', b'\xff\xd8' in chunk, f'chunk={len(chunk)}')

    # 9) Reject bad secret
    r = c.request('POST', '/api/edge/frame', json={'frame': base64.b64encode(jpeg).decode()})
    check('rejects missing secret', r.status_code == 401)

    # 10) Heavy routes return 501 on cloud
    r = c.request('POST', '/api/camera/start', json={'source': 0})
    check('camera/start is edge-only (501)', r.status_code == 501, str(r.json))

    print('=' * 60)
    if fails:
        print(f' RESULT: {fails} check(s) failed')
        return 1
    print(' RESULT: all checks passed')
    print(' Login: demo@aicctv.com / demo123')
    print('=' * 60)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
