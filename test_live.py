#!/usr/bin/env python3
"""
Laptop / CI integration test against a RUNNING cloud server.

  Terminal 1:  cd cloud && python app.py
  Terminal 2:  cd cloud && python test_live.py

Or: python test_live.py --base http://127.0.0.1:5000
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def jpeg_bytes(label: str) -> bytes:
    from PIL import Image, ImageDraw
    img = Image.new('RGB', (480, 270), (25, 35, 50))
    d = ImageDraw.Draw(img)
    d.rectangle([20, 20, 460, 250], outline=(40, 200, 90), width=3)
    d.text((40, 40), f'SIMULATED SERVER — {label}', fill=(255, 255, 255))
    d.text((40, 70), time.strftime('%H:%M:%S'), fill=(180, 220, 255))
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=75)
    return buf.getvalue()


def req(base: str, method: str, path: str, headers=None, body=None, timeout=10):
    url = base.rstrip('/') + path
    data = None
    hdrs = dict(headers or {})
    if body is not None:
        if isinstance(body, (dict, list)):
            data = json.dumps(body).encode()
            hdrs.setdefault('Content-Type', 'application/json')
        else:
            data = body
    r = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            raw = resp.read()
            return resp.status, dict(resp.headers), raw
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--base', default=os.getenv('CLOUD_URL', 'http://127.0.0.1:5000'))
    p.add_argument('--secret', default=os.getenv('EDGE_SYNC_SECRET', 'change-me-edge-sync'))
    args = p.parse_args()
    base, secret = args.base, args.secret
    fails = 0

    def check(name, ok, detail=''):
        nonlocal fails
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f' — {detail}' if detail else ''))
        if not ok:
            fails += 1

    print('=' * 60)
    print(f' Live test → {base}')
    print('=' * 60)

    code, _, raw = req(base, 'GET', '/api/health')
    check('health', code == 200, raw[:120].decode(errors='replace'))

    code, _, raw = req(base, 'GET', '/')
    check('website / (static UI)', code == 200 and b'html' in raw.lower(), f'{len(raw)} bytes')

    code, _, raw = req(base, 'POST', '/api/auth/login',
                       body={'email': 'demo@aicctv.com', 'password': 'demo123'})
    token = None
    try:
        token = json.loads(raw).get('access_token')
    except Exception:
        pass
    check('login demo user', code == 200 and bool(token), raw[:160].decode(errors='replace'))

    auth = {'Authorization': f'Bearer {token}'} if token else {}

    for path in ('/api/stats', '/api/inventory', '/api/trucks', '/api/detections?limit=5', '/api/alerts?limit=5'):
        code, _, raw = req(base, 'GET', path, headers=auth)
        check(path, code == 200, f'status={code}')

    # Simulate Windows server pushing a frame + detections
    jpeg = jpeg_bytes('laptop')
    code, _, raw = req(base, 'POST', '/api/edge/frame',
                       headers={'X-Edge-Secret': secret},
                       body={'frame': base64.b64encode(jpeg).decode()})
    check('simulate server frame push', code == 200, raw.decode(errors='replace')[:120])

    code, hdrs, raw = req(base, 'GET', '/api/video_feed/snapshot')
    check('PA relay snapshot', code == 200 and raw[:2] == b'\xff\xd8', f'{len(raw)} bytes')

    code, _, raw = req(base, 'POST', '/api/edge/sync',
                       headers={'X-Edge-Secret': secret},
                       body={
                           'detections': [
                               {'type': 'sugar_bag', 'direction': 'IN', 'confidence': 0.91, 'count': 1},
                               {'type': 'sugar_bag', 'direction': 'OUT', 'confidence': 0.87, 'count': 1},
                           ],
                           'session': {'session_in': 4, 'session_out': 2},
                       })
    check('simulate server detection sync', code == 200, raw.decode(errors='replace')[:160])

    code, _, raw = req(base, 'GET', '/api/camera/status', headers=auth)
    try:
        body = json.loads(raw)
    except Exception:
        body = {}
    check('relay active', code == 200 and body.get('active') is True, str(body))

    code, _, raw = req(base, 'POST', '/api/camera/start', headers=auth, body={'source': 0})
    check('camera/start blocked on cloud (501)', code == 501)

    print('=' * 60)
    if fails:
        print(f' RESULT: {fails} failed')
        return 1
    print(' RESULT: all live checks passed')
    print(f' Open browser: {base}')
    print(' Login: demo@aicctv.com / demo123')
    print('=' * 60)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
