# v3
"""
Poster Mockup HTTP Service
==========================
FastAPI service that renders one mockup per request and returns JPEG binary.
The poster is fetched from a URL and cached in RAM for 15 minutes so all five
template renders only download the source once (~35 MB).

Endpoints
---------
GET /render?url=<poster_url>&t=<template>&w=2000&q=92&gain=0.5&key=<token>
GET /render?fileid=<drive_file_id>&t=<template>&w=2000&q=92&gain=0.5&key=<token>
    Returns image/jpeg of the rendered mockup. Provide either `url` or
    `fileid` (Google Drive file ID); when `fileid` is set the service
    builds the Drive download URL internally to avoid `&`-escaping issues.

GET /templates?key=<token>
    Returns JSON list of available template names.

GET /health
    Liveness probe — no auth required.

Environment
-----------
MOCKUP_TOKEN   Shared secret required in ?key= on /render and /templates.
               Omit (or leave empty) to disable auth during local dev.
MOCKUP_CACHE   Number of poster images held in RAM (default: 6).
PORT           Listening port (default: 8080).
"""

import io
import ipaddress
import json
import os
import socket
import time
from collections import OrderedDict
from urllib.parse import urlparse

from typing import Optional

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
from PIL import Image

from mockup import _default_quads_path, _default_templates_dir, _load_template, render_mockup

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MOCKUP_TOKEN      = os.environ.get("MOCKUP_TOKEN", "")
MOCKUP_CACHE_SIZE = int(os.environ.get("MOCKUP_CACHE", "6"))
TEMPLATES_DIR     = _default_templates_dir()
QUADS_PATH        = _default_quads_path()

# ---------------------------------------------------------------------------
# Load templates at startup (saves ~0.5 s per request)
# ---------------------------------------------------------------------------
with open(QUADS_PATH) as f:
    QUADS: dict = json.load(f)

TEMPLATES: dict[str, np.ndarray] = {}
for _name in QUADS:
    _path = os.path.join(TEMPLATES_DIR, f"{_name}.png")
    if os.path.exists(_path):
        TEMPLATES[_name] = _load_template(_path)
    else:
        print(f"[warn] template image not found: {_path}")

print(f"[startup] {len(TEMPLATES)} templates loaded: {list(TEMPLATES)}")

# ---------------------------------------------------------------------------
# Poster cache  (LRU + TTL)
# ---------------------------------------------------------------------------
class _PosterCache:
    def __init__(self, maxsize: int = 6, ttl: int = 900):
        self._cache: OrderedDict[str, tuple[bytes, float]] = OrderedDict()
        self.maxsize = maxsize
        self.ttl = ttl

    def get(self, url: str) -> bytes | None:
        if url in self._cache:
            data, ts = self._cache[url]
            if time.time() - ts < self.ttl:
                self._cache.move_to_end(url)
                return data
            del self._cache[url]
        return None

    def put(self, url: str, data: bytes) -> None:
        if url in self._cache:
            self._cache.move_to_end(url)
        else:
            if len(self._cache) >= self.maxsize:
                self._cache.popitem(last=False)
        self._cache[url] = (data, time.time())

    @property
    def size(self) -> int:
        return len(self._cache)


_cache = _PosterCache(maxsize=MOCKUP_CACHE_SIZE)

# ---------------------------------------------------------------------------
# SSRF guard
# ---------------------------------------------------------------------------
_PRIVATE_NETS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]


def _check_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError("Only https:// URLs are accepted")
    host = parsed.hostname or ""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        raise ValueError(f"Cannot resolve host '{host}': {e}")
    for _family, _type, _proto, _canon, sockaddr in infos:
        ip = ipaddress.ip_address(sockaddr[0])
        if ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise ValueError(f"Blocked address: {ip}")
        for net in _PRIVATE_NETS:
            if ip in net:
                raise ValueError(f"Blocked private address: {ip}")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Poster Mockup Service", version="1.0.0")


def _require_auth(key: str) -> None:
    if MOCKUP_TOKEN and key != MOCKUP_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing key")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "templates": list(TEMPLATES.keys()),
        "cached_posters": _cache.size,
    }


@app.get("/templates")
def list_templates(key: str = Query(default="")):
    _require_auth(key)
    return {"templates": list(QUADS.keys())}


@app.get("/render")
async def render(
    t: str = Query(...),
    key: str = Query(...),
    w: int = Query(default=2000),
    q: int = Query(default=92),
    url: Optional[str] = Query(default=None),
    fileid: Optional[str] = Query(default=None),
    gain: Optional[float] = Query(default=None, description="Override reflection gain"),
):
    _require_auth(key)

    if fileid and not url:
        url = f"https://drive.google.com/uc?export=download&id={fileid}&confirm=t"
    if not url:
        raise HTTPException(status_code=400, detail="url or fileid required")

    if t not in QUADS:
        raise HTTPException(status_code=404, detail=f"Unknown template: {t!r}")
    if t not in TEMPLATES:
        raise HTTPException(status_code=503, detail=f"Template image not loaded: {t!r}")

    # SSRF guard
    try:
        _check_url(url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Fetch poster (cache hit avoids re-download for all 5 templates)
    poster_bytes = _cache.get(url)
    if poster_bytes is None:
        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                resp = await client.get(url)
            resp.raise_for_status()
            poster_bytes = resp.content
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=502,
                                detail=f"Poster fetch failed: {e.response.status_code}")
        except Exception as e:
            raise HTTPException(status_code=502,
                                detail=f"Poster fetch error: {e}")
        _cache.put(url, poster_bytes)

    poster_img = Image.open(io.BytesIO(poster_bytes))

    # Downsample source to 2000px max — print res wastes RAM on mockup renders
    MAX_SRC = 2000
    if max(poster_img.size) > MAX_SRC:
        ratio = MAX_SRC / max(poster_img.size)
        poster_img = poster_img.resize(
            (int(poster_img.size[0] * ratio), int(poster_img.size[1] * ratio)),
            Image.LANCZOS
        )

    cfg      = QUADS[t]
    gain_val = gain if gain is not None else float(cfg.get("gain", 1.0))
    clear    = cfg.get("clear", None)

    # Support both "quad" (single) and "quads" (multi-quad, e.g. 4-frame composite)
    quad_data = cfg.get("quads") or [cfg["quad"]]

    # Render (CPU-bound — fine at this volume; add ProcessPoolExecutor if needed)
    result = render_mockup(TEMPLATES[t], poster_img, quad_data,
                           gain=gain_val, clear_rect=clear)

    # Resize
    aspect = result.size[1] / result.size[0]
    out_h  = int(w * aspect)
    result = result.resize((w, out_h), Image.LANCZOS)

    # Encode
    buf = io.BytesIO()
    result.save(buf, "JPEG", quality=q, optimize=True)
    return Response(content=buf.getvalue(), media_type="image/jpeg")


@app.get("/resize")
async def resize_image(
    key: str = Query(...),
    w: int = Query(default=1500),
    q: int = Query(default=88),
    url: Optional[str] = Query(default=None),
    fileid: Optional[str] = Query(default=None),
):
    _require_auth(key)
    if fileid and not url:
        url = f"https://drive.google.com/uc?export=download&id={fileid}&confirm=t"
    if not url:
        raise HTTPException(status_code=400, detail="url or fileid required")
    try:
        _check_url(url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    img_bytes = _cache.get(url)
    if img_bytes is None:
        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                resp = await client.get(url)
            resp.raise_for_status()
            img_bytes = resp.content
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=502, detail=f"Image fetch failed: {e.response.status_code}")
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Image fetch error: {e}")
        _cache.put(url, img_bytes)
    buf_in = io.BytesIO(img_bytes)
    img = Image.open(buf_in)
    img.draft("RGB", (w * 2, w * 2))
    img.load()
    if max(img.size) > w:
        ratio = w / max(img.size)
        img = img.resize((int(img.size[0] * ratio), int(img.size[1] * ratio)), Image.LANCZOS)
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf_out = io.BytesIO()
    img.save(buf_out, "JPEG", quality=q, optimize=True)
    return Response(content=buf_out.getvalue(), media_type="image/jpeg")
# deploy-trigger
