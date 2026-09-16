"""
Poster Mockup HTTP Service
==========================
GET /render?fileid=<drive_id>&t=<template>&key=<token>&w=2000&q=92
GET /render?url=<https_url>&t=<template>&key=<token>&w=2000&q=92
GET /templates?key=<token>
GET /health

Environment vars:
  MOCKUP_TOKEN   — shared secret for /render and /templates (leave empty to disable auth)
  MOCKUP_CACHE   — number of posters held in RAM (default 6)
  PORT           — HTTP port (default 8080)
"""

import io
import os
import time
import ipaddress
import socket
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response

import mockup as _mockup

# ── Config ───────────────────────────────────────────────────────────────────
TOKEN      = os.environ.get("MOCKUP_TOKEN", "")
CACHE_SIZE = int(os.environ.get("MOCKUP_CACHE", "6"))

# ── Simple TTL cache (poster_url → (bgra_array, fetched_at)) ─────────────────
_cache: dict[str, tuple[np.ndarray, float]] = {}

PRIVATE_RANGES = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
]


def _is_private(host: str) -> bool:
    try:
        addr = ipaddress.ip_address(socket.gethostbyname(host))
        return any(addr in net for net in PRIVATE_RANGES)
    except Exception:
        return False


def _safe_url(raw_url: str) -> str:
    """Validate URL: must be https, must not resolve to a private address."""
    parsed = urlparse(raw_url)
    if parsed.scheme != "https":
        raise ValueError("Only https URLs are accepted")
    if _is_private(parsed.hostname or ""):
        raise ValueError("URL resolves to a private/reserved address (SSRF guard)")
    return raw_url


def _fetch_poster(poster_url: str) -> np.ndarray:
    """Fetch poster from URL, decode to BGRA, cache for 15 minutes."""
    now = time.time()
    if poster_url in _cache:
        arr, ts = _cache[poster_url]
        if now - ts < 900:
            return arr

    resp = requests.get(poster_url, timeout=30, stream=True)
    resp.raise_for_status()
    raw = b"".join(resp.iter_content(65536))

    arr_np = np.frombuffer(raw, dtype=np.uint8)
    img    = cv2.imdecode(arr_np, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Image decode failed — unsupported format or corrupt file")

    # Normalise to BGRA
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    # Already 4 channels → keep as-is

    # Evict oldest when cache is full
    if len(_cache) >= CACHE_SIZE:
        oldest = min(_cache, key=lambda k: _cache[k][1])
        del _cache[oldest]

    _cache[poster_url] = (img, now)
    return img


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Poster Mockup Service")


def _check_key(key: str):
    if TOKEN and key != TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")


@app.get("/health")
def health():
    return {"status": "ok", "cache_entries": len(_cache)}


@app.get("/templates")
def templates(key: str = ""):
    _check_key(key)
    return {"templates": list(_mockup.QUADS.keys())}


@app.get("/render")
def render(
    t:      str   = Query(..., description="Template name"),
    key:    str   = Query("",  description="Auth token"),
    url:    str   = Query("",  description="Poster HTTPS URL"),
    fileid: str   = Query("",  description="Google Drive file ID (alternative to url)"),
    w:      int   = Query(2000, description="Output width in pixels"),
    q:      int   = Query(92,   description="JPEG quality 1–100"),
    gain:   float = Query(None, description="Reflection gain override"),
):
    _check_key(key)

    # ── Resolve poster URL ──────────────────────────────────────────────────
    if fileid:
        poster_url = f"https://drive.google.com/uc?export=download&id={fileid}"
    elif url:
        try:
            poster_url = _safe_url(url)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    else:
        raise HTTPException(status_code=400, detail="Provide 'url' or 'fileid'")

    # ── Fetch poster ────────────────────────────────────────────────────────
    try:
        poster = _fetch_poster(poster_url)
    except requests.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Upstream fetch error: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not load poster: {e}")

    # ── Validate template ───────────────────────────────────────────────────
    if t not in _mockup.QUADS:
        raise HTTPException(status_code=404, detail=f"Unknown template '{t}'. "
                            f"Available: {list(_mockup.QUADS.keys())}")

    # ── Render ──────────────────────────────────────────────────────────────
    try:
        result_bgr = _mockup.render(poster, t, gain=gain)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Render error: {e}")

    # ── Scale output to requested width (preserves full aspect ratio) ────────
    rh, rw = result_bgr.shape[:2]
    if w > 0 and w != rw:
        new_h = int(round(rh * w / rw))
        result_bgr = cv2.resize(result_bgr, (w, new_h), interpolation=cv2.INTER_AREA)

    # ── Encode JPEG ─────────────────────────────────────────────────────────
    q = max(1, min(100, q))
    ok, buf = cv2.imencode(".jpg", result_bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
    if not ok:
        raise HTTPException(status_code=500, detail="JPEG encoding failed")

    return Response(content=buf.tobytes(), media_type="image/jpeg")
