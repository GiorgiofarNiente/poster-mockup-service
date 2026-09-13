"""
Poster Mockup Renderer
======================
Composites a poster image into a framed scene using:
  1. Perspective warp  — maps the poster onto the frame's quad
  2. Lighting transfer — applies the template's shadows and glass reflections
  3. Antialiased mask  — 4× supersample edge for clean frame boundary

Supports single-quad templates (one frame) and multi-quad templates (e.g. the
4-frame marketing composite where the poster is warped into all 4 openings).

Usage (CLI):
    python mockup.py poster.png --out out/ --width 2000
    python mockup.py poster.png --only 05_sideboard --gain 0.3
"""

import os
import io
import json
import argparse

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Core rendering
# ---------------------------------------------------------------------------

def _make_mask(quad, height, width, supersample=4):
    """Antialiased polygon mask via 4× supersampling."""
    H4, W4 = height * supersample, width * supersample
    quad4 = (np.array(quad, dtype=np.float32) * supersample).astype(np.int32)
    mask4 = np.zeros((H4, W4), dtype=np.float32)
    cv2.fillPoly(mask4, [quad4.reshape(-1, 1, 2)], 1.0)
    mask = cv2.resize(mask4, (width, height), interpolation=cv2.INTER_AREA)
    return mask[:, :, np.newaxis]   # (H, W, 1)


def _load_template(path):
    """Load template as float32 RGB [0,1]."""
    img = Image.open(path).convert("RGBA")
    return np.array(img, dtype=np.float32) / 255.0   # (H, W, 4)


def render_mockup(template_arr, poster_img, quads,
                  gain=1.0, clear_rect=None):
    """
    Parameters
    ----------
    template_arr : np.ndarray, float32 (H, W, 4)  — template loaded with _load_template()
    poster_img   : PIL.Image.Image  — the poster artwork (any size/mode)
    quads        : list of 4 [x,y] pairs TL→TR→BR→BL  (single quad)
                   OR list of such lists               (multi-quad, e.g. 4-frame composite)
    gain         : float — additive reflection strength (default 1.0; 0.5 for glass-heavy)
    clear_rect   : [x0, y0, x1, y1] — area to wipe white before reading lighting
                   (used for spec-card template to remove placeholder outline)

    Returns
    -------
    PIL.Image.Image  — RGB composite at template resolution
    """
    # Normalise: single quad → list of one quad
    if quads and isinstance(quads[0][0], (int, float)):
        quad_list = [quads]
    else:
        quad_list = list(quads)

    TH, TW = template_arr.shape[:2]

    # Work on RGB float copy
    template_rgb = template_arr[:, :, :3].copy()

    # Wipe placeholder area BEFORE reading lighting (once, before any quad)
    if clear_rect is not None:
        x0, y0, x1, y1 = [int(v) for v in clear_rect]
        template_rgb[y0:y1, x0:x1] = 1.0

    # Pre-compute the poster warp source corners (same for all quads)
    poster_rgb = np.array(poster_img.convert("RGB"), dtype=np.float32) / 255.0
    PH, PW = poster_rgb.shape[:2]
    src_pts = np.float32([[0, 0], [PW, 0], [PW, PH], [0, PH]])

    # Accumulate compositing into result (start from template)
    result = template_rgb.copy()

    for quad in quad_list:
        # --- Lighting extraction for this quad ---
        # Bounding box of the quad for the base-percentile sample
        quad_np = np.array(quad, dtype=np.int32)
        qx0 = max(0, int(quad_np[:, 0].min()))
        qy0 = max(0, int(quad_np[:, 1].min()))
        qx1 = min(TW, int(quad_np[:, 0].max()))
        qy1 = min(TH, int(quad_np[:, 1].max()))
        region = template_rgb[qy0:qy1, qx0:qx1].reshape(-1, 3)

        # 60th-percentile per channel = "unlit paper white"
        base = np.percentile(region, 60, axis=0).astype(np.float32)  # (3,)

        # Per-pixel lighting maps across entire template
        multiply = np.clip(template_rgb / (base[np.newaxis, np.newaxis] + 1e-7), 0.0, 1.0)
        add      = np.clip(template_rgb - base[np.newaxis, np.newaxis],           0.0, 1.0)

        # --- Perspective warp ---
        dst_pts = np.float32(quad)          # TL, TR, BR, BL
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(poster_rgb, M, (TW, TH),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=0)

        # --- Composite into this quad ---
        composited = np.clip(warped * multiply + add * gain, 0.0, 1.0)
        mask = _make_mask(quad, TH, TW)     # (H, W, 1)

        # Blend: poster composite inside quad, accumulated result outside
        result = composited * mask + result * (1.0 - mask)

    return Image.fromarray((result * 255).astype(np.uint8), "RGB")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _default_templates_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")


def _default_quads_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "quads.json")


def cli_main():
    parser = argparse.ArgumentParser(description="Render poster mockups")
    parser.add_argument("poster", help="Path to poster image")
    parser.add_argument("--out",   default="out",   help="Output directory (default: out/)")
    parser.add_argument("--width", type=int, default=2000,
                        help="Output width in pixels (default: 2000)")
    parser.add_argument("--quality", type=int, default=92,
                        help="JPEG quality 1-95 (default: 92)")
    parser.add_argument("--only",  default=None,
                        help="Render only this template name, e.g. 05_sideboard")
    parser.add_argument("--gain",  type=float, default=None,
                        help="Override reflection gain (template default otherwise)")
    args = parser.parse_args()

    quads_path     = _default_quads_path()
    templates_dir  = _default_templates_dir()

    with open(quads_path) as f:
        quads = json.load(f)

    poster = Image.open(args.poster)
    os.makedirs(args.out, exist_ok=True)

    for name, cfg in quads.items():
        if args.only and name != args.only:
            continue

        template_path = os.path.join(templates_dir, f"{name}.png")
        if not os.path.exists(template_path):
            print(f"  [skip] template not found: {template_path}")
            continue

        template = _load_template(template_path)
        gain_val  = args.gain if args.gain is not None else float(cfg.get("gain", 1.0))
        clear     = cfg.get("clear", None)

        # Support both "quad" (single) and "quads" (multi-quad)
        quad_data = cfg.get("quads") or [cfg["quad"]]

        print(f"  Rendering {name} …", end=" ", flush=True)
        result = render_mockup(template, poster, quad_data,
                               gain=gain_val, clear_rect=clear)

        # Resize to requested width
        aspect = result.size[1] / result.size[0]
        out_h  = int(args.width * aspect)
        result = result.resize((args.width, out_h), Image.LANCZOS)

        out_path = os.path.join(args.out, f"{name}.jpg")
        result.save(out_path, "JPEG", quality=args.quality, optimize=True)
        print(f"→ {out_path}")


if __name__ == "__main__":
    cli_main()
