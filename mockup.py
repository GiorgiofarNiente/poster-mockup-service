"""
Poster mockup renderer.
Composites a poster onto a scene template using perspective warp + lighting transfer.

Key behaviour:
- If the poster's aspect ratio is wider than the frame's, the poster is center-cropped
  left/right (top and bottom are NEVER touched) so it matches the frame's aspect ratio
  before being warped in.  This handles e.g. a 3:4 poster going into a 2:3 frame.
- Full template image is always output (no cropping of the scene).
- Templates with "quads" (plural, e.g. 01_specs 4-frame composite) render the poster
  into every frame.
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
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def _crop_poster(poster_bgra: np.ndarray, quad: np.ndarray):
    """Center-crop poster left/right to match frame aspect ratio. Never touches top/bottom."""
    tl, tr, br, bl = quad
    frame_w      = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2.0
    frame_h      = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0
    frame_aspect = frame_w / frame_h

    ph, pw        = poster_bgra.shape[:2]
    poster_aspect = pw / ph

    if poster_aspect > frame_aspect + 1e-4:
        new_pw  = max(int(round(ph * frame_aspect)), 1)
        x_start = (pw - new_pw) // 2
        poster_bgra = poster_bgra[:, x_start : x_start + new_pw]

    return poster_bgra


def _composite_quad(tmpl_f: np.ndarray, poster_bgra: np.ndarray,
                    quad: np.ndarray, g: float) -> np.ndarray:
    """
    Warp poster into one quad on tmpl_f (float32 BGR H×W×3) and return
    an updated float32 BGR image.
    """
    th, tw = tmpl_f.shape[:2]

    cropped   = _crop_poster(poster_bgra, quad)
    ph, pw    = cropped.shape[:2]

    src_pts = np.array([
        [0,      0     ],
        [pw - 1, 0     ],
        [pw - 1, ph - 1],
        [0,      ph - 1],
    ], dtype=np.float32)

    dst_pts = quad

    M        = cv2.getPerspectiveTransform(src_pts, dst_pts)
    poster_f = cropped[:, :, :3].astype(np.float32) / 255.0
    warped   = cv2.warpPerspective(poster_f, M, (tw, th),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=(0, 0, 0))

    # Antialiased quad mask (4x supersample)
    scale    = 4
    big_mask = np.zeros((th * scale, tw * scale), dtype=np.uint8)
    cv2.fillPoly(big_mask, [(dst_pts * scale).astype(np.int32)], 255)
    mask_small = cv2.resize(big_mask, (tw, th), interpolation=cv2.INTER_AREA)
    mask_f     = mask_small[:, :, np.newaxis].astype(np.float32) / 255.0

    # Lighting transfer
    art_pixels = tmpl_f[mask_small > 127]
    if len(art_pixels) > 0:
        base = np.percentile(art_pixels, 60, axis=0).reshape(1, 1, 3)
    else:
        base = np.array([[[1., 1., 1.]]], dtype=np.float32)

    multiply = np.clip(tmpl_f / (base + 1e-7), 0., 1.)
    add      = np.clip(tmpl_f - base,          0., 1.)

    lit      = np.clip(warped * multiply + add * g, 0., 1.)

    return tmpl_f * (1. - mask_f) + lit * mask_f


def render(poster_bgra: np.ndarray, template_name: str, gain: float = None) -> np.ndarray:
    """
    Composite poster_bgra into the named template.
    Returns a BGR uint8 array (full template dimensions).
    """
    cfg = QUADS[template_name]
    g   = gain if gain is not None else float(cfg.get("gain", 1.0))

    # Normalise: support both "quad" (single) and "quads" (multi-frame)
    if "quad" in cfg:
        quad_list = [np.array(cfg["quad"], dtype=np.float32)]
    elif "quads" in cfg:
        quad_list = [np.array(q, dtype=np.float32) for q in cfg["quads"]]
    else:
        raise KeyError(f"Template '{template_name}' has no 'quad' or 'quads' key")

    tmpl   = _load_template(template_name).copy()
    th, tw = tmpl.shape[:2]

    if "clear" in cfg:
        x0, y0, x1, y1 = cfg["clear"]
        tmpl[y0:y1, x0:x1] = [255, 255, 255, 255]

    result_f = tmpl[:, :, :3].astype(np.float32) / 255.0

    for quad in quad_list:
        result_f = _composite_quad(result_f, poster_bgra, quad, g)

    return (np.clip(result_f, 0., 1.) * 255.).astype(np.uint8)
