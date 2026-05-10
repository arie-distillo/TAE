"""
core/segment_store.py — SAM2 segment persistence
==================================================
One SQLite database per mission (missions/{id}/segments.db).

Each row represents one SAM2-generated region from a full drone frame:
  - frame_path  : absolute path to the parent frame (links back to uploads)
  - bbox        : [xmin, ymin, xmax, ymax] in full-frame pixel coordinates
  - area        : region area in pixels² (used for noise/background filtering)
  - clip_vector : 512-dim CLIP embedding of the region crop (512×float32 blob)

At query time, all segments for frames retrieved by CLIP ANN search are
loaded and scored against a vocabulary — pure numpy dot products, no model
inference needed.  This is the payoff for doing the SAM2+CLIP work at ingest.
"""

import json
import logging
import struct
import sqlite3
from pathlib import Path

import numpy as np

logger = logging.getLogger("TAE.SegmentStore")

# Segments smaller than this are noise (< 20×20 px)
MIN_SEGMENT_AREA_PX: int   = 400

# Segments larger than this fraction of the frame are dominant background
# (ocean, crop field, tarmac) and carry no anomaly signal by themselves
MAX_SEGMENT_AREA_FRAC: float = 0.40


class SegmentStore:
    """
    SQLite-backed store for SAM2 segments and their CLIP region vectors.

    Design choices:
    - clip_vector stored as a raw binary BLOB (512 × float32 little-endian).
      struct.pack/unpack is ~10× faster than json and produces a 2KB record
      vs ~12KB JSON, keeping the DB compact and reads fast.
    - One index on frame_path so per-frame retrieval is O(log N) not O(N).
    - has_frame() allows idempotent re-runs: if a frame was already segmented
      (e.g. from a previous partial ingestion), it is skipped without error.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS segments (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    frame_path   TEXT    NOT NULL,
                    bbox_json    TEXT    NOT NULL,  -- [xmin,ymin,xmax,ymax] full-frame px
                    area         INTEGER NOT NULL,
                    clip_vector  BLOB    NOT NULL   -- 512×float32 little-endian
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_segments_frame "
                "ON segments(frame_path)"
            )
            conn.commit()

    @staticmethod
    def _pack_vector(vec: np.ndarray) -> bytes:
        arr = vec.flatten().astype(np.float32)
        return struct.pack(f"{len(arr)}f", *arr)

    @staticmethod
    def _unpack_vector(blob: bytes) -> np.ndarray:
        n = len(blob) // 4
        return np.array(struct.unpack(f"{n}f", blob), dtype=np.float32)

    # ── Public API ────────────────────────────────────────────────────────────

    def has_frame(self, frame_path: str | Path) -> bool:
        """True if at least one segment exists for this frame."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM segments WHERE frame_path = ? LIMIT 1",
                (str(frame_path),),
            ).fetchone()
        return row is not None

    def insert_segments(
        self,
        frame_path: str | Path,
        segments:   list[dict],
    ) -> int:
        """
        Bulk-insert all segments for one frame in a single transaction.

        Each dict in `segments` must contain:
            bbox        : [xmin, ymin, xmax, ymax]  (full-frame pixels)
            area        : int
            clip_vector : np.ndarray shape (512,)

        Returns the number of rows inserted.
        """
        if not segments:
            return 0

        rows = [
            (
                str(frame_path),
                json.dumps(seg["bbox"]),
                int(seg["area"]),
                self._pack_vector(seg["clip_vector"]),
            )
            for seg in segments
        ]
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO segments (frame_path, bbox_json, area, clip_vector) "
                "VALUES (?, ?, ?, ?)",
                rows,
            )
            conn.commit()
        return len(rows)

    def get_segments(self, frame_path: str | Path) -> list[dict]:
        """
        Returns all stored segments for a frame.

        Each returned dict contains:
            bbox        : [xmin, ymin, xmax, ymax]
            area        : int
            clip_vector : np.ndarray shape (512,)
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT bbox_json, area, clip_vector "
                "FROM segments WHERE frame_path = ?",
                (str(frame_path),),
            ).fetchall()

        return [
            {
                "bbox":        json.loads(r["bbox_json"]),
                "area":        r["area"],
                "clip_vector": self._unpack_vector(r["clip_vector"]),
            }
            for r in rows
        ]

    def get_segments_for_frames(
        self, frame_paths: list[str]
    ) -> dict[str, list[dict]]:
        """
        Bulk retrieval: returns {frame_path: [segments]} for a list of frames.
        More efficient than calling get_segments() in a loop.
        """
        if not frame_paths:
            return {}
        placeholders = ",".join("?" * len(frame_paths))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT frame_path, bbox_json, area, clip_vector "
                f"FROM segments WHERE frame_path IN ({placeholders})",
                frame_paths,
            ).fetchall()

        result: dict[str, list[dict]] = {fp: [] for fp in frame_paths}
        for r in rows:
            result[r["frame_path"]].append({
                "bbox":        json.loads(r["bbox_json"]),
                "area":        r["area"],
                "clip_vector": self._unpack_vector(r["clip_vector"]),
            })
        return result

    # ── Stats ─────────────────────────────────────────────────────────────────

    def frame_count(self) -> int:
        with self._connect() as conn:
            return conn.execute(
                "SELECT COUNT(DISTINCT frame_path) FROM segments"
            ).fetchone()[0]

    def segment_count(self) -> int:
        with self._connect() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM segments"
            ).fetchone()[0]
