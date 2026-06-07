"""
core/tile_novelty.py — Tile novelty tracker for streaming pipeline.

Classifies tiles as 'foreground' (new geographic content) or 'background'
(already-seen area) to prioritise detection on novel tiles for near-real-time
responsiveness.

Adaptive threshold
------------------
The overlap threshold is derived from the drone's actual movement relative
to tile size.  Each frame, the tracker computes:

    shift_frac = EMA(drone_shift_m) / tile_width_m

and sets:

    overlap_threshold = 1 − max(MIN_NOVELTY, 0.5 × shift_frac)

This ensures that at least the leading-edge tiles qualify as foreground
after a single frame of movement, regardless of drone speed or altitude.
Slow flight → high threshold (~0.90), fast flight → lower (~0.60).

Track-aware promotion
---------------------
Tiles covering the predicted position of an active track are always
promoted to foreground, regardless of overlap.

Implementation
--------------
Uses a discretised geographic grid (~3 m cells) for fast set-intersection
overlap computation.  No external geo-libraries required.

BackgroundDetectionWorker
-------------------------
Daemon-thread consumer that drains a queue of 'seen' tiles and runs
detection at lower priority, so background tiles are eventually
processed without blocking the foreground path.
"""

from __future__ import annotations

import logging
import math
import queue
import threading
from typing import Callable

logger = logging.getLogger("TAE.Novelty")

# ── tunables ──────────────────────────────────────────────────────────────────
OVERLAP_THRESHOLD = 0.60   # initial / fallback — overridden by adaptive logic
GRID_CELL_M       = 3.0    # grid cell size in metres

# Adaptive threshold bounds
MIN_NOVELTY       = 0.10   # always classify tiles with ≥10 % new area as FG
MAX_THRESHOLD     = 0.95   # never require >95 % overlap to call a tile "seen"
MIN_THRESHOLD     = 0.50   # never classify tiles with <50 % overlap as "seen"
EMA_ALPHA         = 0.4    # EMA smoothing for drone shift (0 = ignore new, 1 = no smoothing)

_M_PER_DEG_LAT = 111_320.0


def _m_per_deg_lon(lat: float) -> float:
    return 111_320.0 * math.cos(math.radians(lat))


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two (lat, lon) points."""
    R = 6_371_000.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ─────────────────────────────────────────────────────────────────────────────
# TileNoveltyTracker
# ─────────────────────────────────────────────────────────────────────────────

class TileNoveltyTracker:
    """
    Fast grid-based tile novelty tracker.

    Maintains a set of discretised geographic cells that have been seen.
    For each incoming tile, rasterises its footprint quadrilateral into
    grid cells and checks what fraction are already in the seen set.
    """

    def __init__(
        self,
        overlap_threshold: float = OVERLAP_THRESHOLD,
        grid_cell_m:       float = GRID_CELL_M,
    ):
        self._threshold = overlap_threshold
        self._cell_m    = grid_cell_m
        self._seen: set[tuple[int, int]] = set()
        self._ref_lat: float | None = None
        self._lat_scale = 0.0
        self._lon_scale = 0.0
        self._tiles_processed = 0
        # Adaptive threshold state
        self._prev_center: tuple[float, float] | None = None
        self._ema_shift_m  = 0.0
        self._tile_width_m = 0.0

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear all state — call when starting a new stream."""
        self._seen.clear()
        self._ref_lat = None
        self._lat_scale = 0.0
        self._lon_scale = 0.0
        self._tiles_processed = 0
        self._prev_center = None
        self._ema_shift_m = 0.0
        self._tile_width_m = 0.0
        self._threshold = OVERLAP_THRESHOLD
        logger.info("Novelty tracker reset")

    # ── grid helpers ──────────────────────────────────────────────────────────

    def _init_grid(self, lat: float) -> None:
        self._ref_lat   = lat
        self._lat_scale = _M_PER_DEG_LAT / self._cell_m
        self._lon_scale = _m_per_deg_lon(lat) / self._cell_m

    def _to_cell(self, lat: float, lon: float) -> tuple[int, int]:
        return (int(lat * self._lat_scale), int(lon * self._lon_scale))

    def _rasterise(self, tile: dict) -> set[tuple[int, int]]:
        """
        Rasterise a tile's quadrilateral footprint to grid cells.

        Uses the axis-aligned bounding box of the four geo-corners.
        Slightly over-counts area (conservative), but fast and never
        under-counts.

        Safety: caps grid dimensions at MAX_CELLS_PER_DIM to prevent
        OOM from bad altitude / footprint data (e.g. 85 km AGL from
        a misparse producing tile footprints spanning degrees).
        """
        MAX_CELLS_PER_DIM = 200   # ~600 m at 3 m resolution — generous for real tiles

        lats = [
            tile.get("fp_nw_lat", 0.0), tile.get("fp_ne_lat", 0.0),
            tile.get("fp_se_lat", 0.0), tile.get("fp_sw_lat", 0.0),
        ]
        lons = [
            tile.get("fp_nw_lon", 0.0), tile.get("fp_ne_lon", 0.0),
            tile.get("fp_se_lon", 0.0), tile.get("fp_sw_lon", 0.0),
        ]

        # Skip tiles without footprint data
        if not any(lats) or not any(lons):
            return set()

        if self._ref_lat is None:
            self._init_grid(sum(lats) / 4.0)

        min_r = int(min(lats) * self._lat_scale)
        max_r = int(max(lats) * self._lat_scale) + 1
        min_c = int(min(lons) * self._lon_scale)
        max_c = int(max(lons) * self._lon_scale) + 1

        rows = max_r - min_r + 1
        cols = max_c - min_c + 1
        if rows > MAX_CELLS_PER_DIM or cols > MAX_CELLS_PER_DIM:
            logger.warning(
                "Tile footprint too large (%d×%d cells, ~%.0fm×%.0fm) "
                "— likely bad altitude; skipping rasterisation",
                rows, cols,
                rows * self._cell_m, cols * self._cell_m,
            )
            return set()

        return {
            (r, c)
            for r in range(min_r, max_r + 1)
            for c in range(min_c, max_c + 1)
        }

    # ── adaptive threshold ───────────────────────────────────────────────────

    def update_threshold(
        self,
        frame_lat: float,
        frame_lon: float,
        tile_width_m: float = 0.0,
    ) -> None:
        """
        Adapt the overlap threshold based on actual drone movement.

        Called once per frame BEFORE classify_tiles().  Computes the
        geographic shift from the previous frame's center, smooths it
        with an EMA, and derives the threshold from the ratio of shift
        to tile width.

        The formula ensures that leading-edge tiles qualify as foreground
        after a single frame of movement:

            shift_frac  = ema_shift / tile_width
            novelty_min = max(MIN_NOVELTY, 0.5 × shift_frac)
            threshold   = clamp(1 − novelty_min, MIN_THRESHOLD, MAX_THRESHOLD)

        Parameters
        ----------
        frame_lat, frame_lon
            Drone position for this frame (from telemetry).
        tile_width_m
            Approximate tile width in metres (from footprint corners).
            Cached after the first non-zero value.
        """
        if tile_width_m > 0:
            self._tile_width_m = tile_width_m

        if self._prev_center is not None and self._tile_width_m > 0:
            # Haversine distance between consecutive frame centres
            lat1, lon1 = self._prev_center
            shift_m = _haversine_m(lat1, lon1, frame_lat, frame_lon)

            # EMA-smooth to absorb GPS jitter
            if self._ema_shift_m == 0.0:
                self._ema_shift_m = shift_m       # seed with first measurement
            else:
                self._ema_shift_m = (
                    EMA_ALPHA * shift_m
                    + (1.0 - EMA_ALPHA) * self._ema_shift_m
                )

            shift_frac  = self._ema_shift_m / self._tile_width_m
            novelty_min = max(MIN_NOVELTY, 0.5 * shift_frac)
            new_thresh  = 1.0 - novelty_min
            self._threshold = max(MIN_THRESHOLD, min(MAX_THRESHOLD, new_thresh))

            logger.info(
                "Adaptive threshold: shift=%.1fm  tile=%.0fm  frac=%.2f  "
                "threshold=%.2f",
                self._ema_shift_m, self._tile_width_m,
                shift_frac, self._threshold,
            )

        self._prev_center = (frame_lat, frame_lon)

    # ── public API ────────────────────────────────────────────────────────────

    def classify_tiles(
        self,
        tiles: list[dict],
        track_positions: list[tuple[float, float]] | None = None,
    ) -> tuple[list[dict], list[dict]]:
        """
        Split tiles into foreground (novel / track-promoted) and background.

        Parameters
        ----------
        tiles
            Tile dicts from LanceDB — must have fp_*_lat/lon fields.
        track_positions
            (lat, lon) predicted positions of active tracks.  Tiles whose
            footprint contains any of these points are promoted to foreground.

        Returns
        -------
        (foreground, background)
        """
        if not tiles:
            return [], []

        # Pre-compute grid cells for active track positions
        track_cells: set[tuple[int, int]] = set()
        if track_positions:
            if self._ref_lat is None:
                t0 = tiles[0]
                lat0 = t0.get("fp_nw_lat") or t0.get("lat", 0.0)
                if lat0:
                    self._init_grid(lat0)
            if self._ref_lat is not None:
                for lat, lon in track_positions:
                    if lat and lon:
                        track_cells.add(self._to_cell(lat, lon))

        foreground: list[dict] = []
        background: list[dict] = []

        for tile in tiles:
            cells = self._rasterise(tile)

            if not cells:
                # No footprint data → play it safe, treat as foreground
                foreground.append(tile)
                continue

            # Track-aware promotion
            if track_cells and (track_cells & cells):
                foreground.append(tile)
                continue

            # Geographic novelty check
            if not self._seen:
                # First frame — everything is novel
                foreground.append(tile)
                continue

            overlap = len(cells & self._seen)
            frac    = overlap / len(cells)

            if frac < self._threshold:
                foreground.append(tile)
            else:
                background.append(tile)

        return foreground, background

    def mark_processed(self, tiles: list[dict]) -> None:
        """
        Register tiles as processed — adds their grid cells to the seen set.
        Call AFTER detection has completed on these tiles.
        """
        for tile in tiles:
            cells = self._rasterise(tile)
            self._seen.update(cells)
            self._tiles_processed += 1

    @property
    def cells_seen(self) -> int:
        return len(self._seen)

    @property
    def tiles_processed(self) -> int:
        return self._tiles_processed


# ─────────────────────────────────────────────────────────────────────────────
# BackgroundDetectionWorker
# ─────────────────────────────────────────────────────────────────────────────

class BackgroundDetectionWorker:
    """
    Daemon thread that processes 'background' (seen) tiles asynchronously.

    Work items are (tiles, callback) tuples.  The callback receives the
    tile list and is responsible for running detection and updating state.
    """

    def __init__(self, max_batch: int = 8):
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._running = False
        self._max_batch = max_batch

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        # Drain any leftover items from a previous session
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="bg-detect",
        )
        self._thread.start()
        logger.info("Background detection worker started")

    def stop(self) -> None:
        self._running = False
        # Unblock the thread if it's waiting on the queue
        self._queue.put(None)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._thread = None
        logger.info("Background detection worker stopped")

    def enqueue(
        self,
        tiles: list[dict],
        callback: Callable[[list[dict]], None],
    ) -> None:
        """
        Enqueue a batch of background tiles for deferred detection.

        Parameters
        ----------
        tiles
            Tile dicts to process.
        callback
            Function that receives the tile list and runs the detection
            pipeline stages, updating state as needed.
        """
        if not self._running:
            logger.debug("BG worker not running — dropping %d tiles", len(tiles))
            return
        self._queue.put((tiles, callback))

    @property
    def pending(self) -> int:
        return self._queue.qsize()

    def _loop(self) -> None:
        while self._running:
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if item is None:
                # Poison pill — exit
                break

            tiles, callback = item
            try:
                callback(tiles)
            except Exception as exc:
                logger.error("BG detection error: %s", exc, exc_info=True)

        logger.info("Background detection worker loop exited")