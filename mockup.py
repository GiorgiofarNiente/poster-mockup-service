"""
Poster mockup renderer.
Composites a poster onto a scene template using perspective warp + lighting transfer.

Key behaviour:
- If the poster's aspect ratio is wider than the frame's, the poster is center-cropped
  left/right (top and bottom are NEVER touched) so it matches the frame's aspect ratio
  before being warped in.  This handles e.g. a 3:4 poster going into a 2:3 frame.
- Full template image is always output (no cropping of the scene).
"""

import json
import cv2
import numpy as np
from pathlib import Path

TEMPLATE_DIR = Path(__file__).parent / "templates"
QUADS_FILE   = Path(__file__).parent / "quads.json"

with open(QUADS_FILE) as f:
    QUADS = json.load(f)


def _load_template(name: str) -> np.ndarray:
    path = TEMPLATE_DIR / f"{name}.png"
    img  = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Template not found: {path}")
    # Ensure 4 channels
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def render(poster_bgra: np.ndarray, template_name: str, gain: float = None) -> np.ndarray:
    """
    Composite poster_bgra into the named template.
    Returns a BGR uint8 array (full template dimensions, scaled to `w` later by server).
    """
    cfg   = QUADS[template_name]
    quad  = np.array(cfg["quad"], dtype=np.float32)  # [TL, TR, BR, BL]
    g     = gain if gain is not None else float(cfg.get("gain", 1.0))

    # ── Load template ────────────────────────────────────────────────────────
    tmpl      = _load_template(template_name).copy()
    th, tw    = tmpl.shape[:2]

    # Clear placeholder region BEFORE reading lighting (prevents placeholder
    # marks from being baked into the shadow/highlight layer).
    if "clear" in cfg:
        x0, y0, x1, y1 = cfg["clear"]
        tmpl[y0:y1, x0:x1] = [255, 255, 255, 255]

    # ── Step 0: Center-crop poster to match frame aspect ratio ───────────────
    # Only crop left/right.  Top and bottom are never touched.
    tl, tr, br, bl = quad
    frame_w      = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2.0
    frame_h      = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0
    frame_aspect = frame_w / frame_h          # e.g. ≈ 0.667 for a 2:3 frame

    ph, pw       = poster_bgra.shape[:2]
    poster_aspect = pw / ph                   # e.g.  0.75 for a 3:4 poster

    if poster_aspect > frame_aspect + 1e-4:
        # Poster is wider than the frame → trim equal strips from left and right
        new_pw  = int(round(ph * frame_aspect))
        new_pw  = max(new_pw, 1)
        x_start = (pw - new_pw) // 2
        poster_bgra = poster_bgra[:, x_start : x_start + new_pw]
        ph, pw  = poster_bgra.shape[:2]
    # If poster is already narrower / equal: no crop (never touch top or bottom)

    # ── Step 1: Perspective-warp poster into the frame quad ─────────────────
    # Map poster corners → quad corners (TL TR BR BL order).
    src_pts = np.array([
        [0,      0     ],
        [pw - 1, 0     ],
        [pw - 1, ph - 1],
        [0,      ph - 1],
    ], dtype=np.float32)

    dst_pts = quad   # positions in template image space

    M      = cv2.getPerspectiveTransform(src_pts, dst_pts)
    poster_f = poster_bgra[:, :, :3].astype(np.float32) / 255.0  # BGR float
    warped   = cv2.warpPerspective(poster_f, M, (tw, th),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=(0, 0, 0))

    # ── Step 2: Antialiased quad mask (4× supersample) ───────────────────────
    scale    = 4
    big_mask = np.zeros((th * scale, tw * scale), dtype=np.uint8)
    big_quad = (dst_pts * scale).astype(np.int32)
    cv2.fillPoly(big_mask, [big_quad], 255)
    mask_small = cv2.resize(big_mask, (tw, th), interpolation=cv2.INTER_AREA)
    mask_f     = mask_small[:, :, np.newaxis].astype(np.float32) / 255.0

    # ── Step 3: Lighting transfer ────────────────────────────────────────────
    # Sample only the artwork region of the (possibly placeholder-cleared) template.
    tmpl_f   = tmpl[:, :, :3].astype(np.float32) / 255.0   # BGR float

    art_bool  = mask_small > 127
    art_pixels = tmpl_f[art_bool]                            # (N, 3)

    if len(art_pixels) > 0:
        base = np.percentile(art_pixels, 60, axis=0)        # per-channel "unlit paper"
    else:
        base = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    base    = base.reshape(1, 1, 3)
    multiply = np.clip(tmpl_f / (base + 1e-7), 0.0, 1.0)   # shadows (darkening)
    add      = np.clip(tmpl_f - base,           0.0, 1.0)   # glass reflections (additive)

    # ── Step 4: Composite ────────────────────────────────────────────────────
    lit       = warped * multiply + add * g
    lit       = np.clip(lit, 0.0, 1.0)

    result_f  = tmpl_f * (1.0 - mask_f) + lit * mask_f
    result_bgr = (np.clip(result_f, 0.0, 1.0) * 255.0).astype(np.uint8)

    return result_bgr
