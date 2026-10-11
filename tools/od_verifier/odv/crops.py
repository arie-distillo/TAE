"""
Crop strategies — how much of the tile the local verifier sees.

    tight<P>     box plus P % padding of its longer side      (tight20 = the
                 10–20 % padding usually recommended for crop verifiers)
    ctx<K>       box plus K × its longer side of context, with a floor of
                 12·K px (ctx4 mirrors TAE's _padded_upscaled_crop: 4×, ≥48 px)
    ...box       same, with the candidate box drawn in (as TAE does for the VLM)

All crops are square windows centred on the box and shifted to stay inside
the tile, so model preprocessing (resize / centre-crop) never cuts the object.
"""
from __future__ import annotations

import re

import cv2
import numpy as np

_SPEC = re.compile(r"^(tight|ctx)(\d+)(box)?$")
_BOX_BGR = (74, 222, 128)      # same green TAE draws for the VLM
_MIN_SIDE = 16


def parse(strategy: str) -> tuple[str, int, bool]:
    m = _SPEC.match(strategy)
    if not m:
        raise ValueError(
            f"unknown crop strategy '{strategy}' — use tight<P>, ctx<K>, optionally +'box' "
            f"(e.g. tight20, ctx2, ctx4box)")
    return m.group(1), int(m.group(2)), bool(m.group(3))


def make_crop(tile: np.ndarray, bbox: list, strategy: str) -> np.ndarray:
    kind, k, draw = parse(strategy)
    th, tw = tile.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, min(x1, tw - 1)); x2 = max(x1 + 1, min(x2, tw))
    y1 = max(0, min(y1, th - 1)); y2 = max(y1 + 1, min(y2, th))
    long_side = max(x2 - x1, y2 - y1)

    pad = long_side * k / 100.0 if kind == "tight" else max(12.0 * k, float(long_side * k))
    side = int(round(long_side + 2 * pad))
    side = max(_MIN_SIDE, min(side, tw, th))

    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    wx = int(round(cx - side / 2.0)); wy = int(round(cy - side / 2.0))
    wx = max(0, min(wx, tw - side)); wy = max(0, min(wy, th - side))

    crop = tile[wy:wy + side, wx:wx + side].copy()
    if draw:
        cv2.rectangle(crop, (x1 - wx, y1 - wy), (x2 - wx - 1, y2 - wy - 1), _BOX_BGR, 1)
    return crop
