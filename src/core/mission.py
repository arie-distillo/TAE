"""
core/mission.py — Mission lifecycle management
================================================
Missions are the top-level organisational unit in TAE.  Each mission wraps:
  - a set of uploaded frames (upload_path)
  - a LanceDB instance (lancedb_path) for CLIP tile vectors
  - a SQLite segment store (segments_db_path) for SAM2 region vectors
  - a declared list of allowed query intents

All missions are catalogued in a single global missions.db SQLite file so
the application can resume an existing mission across restarts without
requiring the operator to re-upload data.

Minimalist startup behaviour (Phase 1):
  - On each startup, look for an existing mission whose upload_path matches
    the configured UPLOAD_PATH.  If found, resume it.  If not, create one.
  - allowed_intents is hardcoded to ["object_search", "anomaly_detection"].
  - Mission creation UI and multi-mission management come in a later phase.
"""

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("TAE.Mission")

# Intents enabled for every auto-created mission until UI allows customisation
ALLOWED_INTENTS_DEFAULT: list[str] = ["object_search", "anomaly_detection"]


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Mission:
    id:               str
    name:             str
    allowed_intents:  list[str]
    upload_path:      str        # absolute path to the frames directory
    lancedb_path:     str        # absolute path to the LanceDB directory
    segments_db_path: str        # absolute path to segments.db
    created_at:       str        # ISO-8601 UTC
    status:           str        # "created" | "ingesting" | "ready"
    frame_count:      int  = 0
    tile_count:       int  = 0
    segment_count:    int  = 0
    scene_context:    str  = ""  # optional operator hint for anomaly queries

    def allows(self, intent: str) -> bool:
        return intent in self.allowed_intents

    @property
    def short_id(self) -> str:
        return self.id[:8]


# ─────────────────────────────────────────────────────────────────────────────
# Manager
# ─────────────────────────────────────────────────────────────────────────────

class MissionManager:
    """
    CRUD over the global missions.db SQLite catalogue.
    One MissionManager instance is created at application startup and kept
    for the lifetime of the process.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS missions (
                    id               TEXT    PRIMARY KEY,
                    name             TEXT    NOT NULL,
                    allowed_intents  TEXT    NOT NULL,   -- JSON array
                    upload_path      TEXT    NOT NULL,
                    lancedb_path     TEXT    NOT NULL,
                    segments_db_path TEXT    NOT NULL,
                    created_at       TEXT    NOT NULL,
                    status           TEXT    NOT NULL DEFAULT 'created',
                    frame_count      INTEGER NOT NULL DEFAULT 0,
                    tile_count       INTEGER NOT NULL DEFAULT 0,
                    segment_count    INTEGER NOT NULL DEFAULT 0,
                    scene_context    TEXT    NOT NULL DEFAULT ''
                )
            """)
            conn.commit()

    @staticmethod
    def _row_to_mission(row: sqlite3.Row) -> Mission:
        return Mission(
            id               = row["id"],
            name             = row["name"],
            allowed_intents  = json.loads(row["allowed_intents"]),
            upload_path      = row["upload_path"],
            lancedb_path     = row["lancedb_path"],
            segments_db_path = row["segments_db_path"],
            created_at       = row["created_at"],
            status           = row["status"],
            frame_count      = row["frame_count"],
            tile_count       = row["tile_count"],
            segment_count    = row["segment_count"],
            scene_context    = row["scene_context"] or "",
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def get_or_create_active(
        self,
        upload_path:      str | Path,
        lancedb_path:     str | Path,
        segments_db_path: str | Path,
        allowed_intents:  list[str] | None = None,
    ) -> Mission:
        """
        Resume the most recent mission for this upload_path, or create one.

        Matching on upload_path means that restarting the application while
        the same data directory is configured will seamlessly resume the
        existing mission — operators don't lose their context.
        """
        upload_str = str(upload_path)
        with self._connect() as conn:
            row = conn.execute(
                """SELECT * FROM missions
                   WHERE upload_path = ?
                   ORDER BY created_at DESC LIMIT 1""",
                (upload_str,),
            ).fetchone()

            if row:
                mission = self._row_to_mission(row)
                logger.info(
                    f"Mission resumed | '{mission.name}' ({mission.short_id}) | "
                    f"intents: {mission.allowed_intents} | "
                    f"status: {mission.status}"
                )
                return mission

            # ── First run for this upload_path — create a new mission ─────────
            intents    = allowed_intents or ALLOWED_INTENTS_DEFAULT
            mission_id = uuid.uuid4().hex
            name       = f"Mission {datetime.now():%Y-%m-%d %H:%M}"
            now        = datetime.utcnow().isoformat()

            conn.execute(
                """INSERT INTO missions
                       (id, name, allowed_intents, upload_path, lancedb_path,
                        segments_db_path, created_at, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 'created')""",
                (
                    mission_id, name,
                    json.dumps(intents),
                    upload_str,
                    str(lancedb_path),
                    str(segments_db_path),
                    now,
                ),
            )
            conn.commit()

        mission = Mission(
            id               = mission_id,
            name             = name,
            allowed_intents  = intents,
            upload_path      = upload_str,
            lancedb_path     = str(lancedb_path),
            segments_db_path = str(segments_db_path),
            created_at       = now,
            status           = "created",
        )
        logger.info(
            f"Mission created | '{name}' ({mission.short_id}) | "
            f"intents: {intents}"
        )
        return mission

    def update_counts(
        self,
        mission_id:    str,
        status:        str,
        frame_count:   int | None = None,
        tile_count:    int | None = None,
        segment_count: int | None = None,
    ) -> None:
        """Update mission status and optional counts after ingestion."""
        with self._connect() as conn:
            if frame_count is not None:
                conn.execute(
                    """UPDATE missions
                       SET status=?, frame_count=?, tile_count=?, segment_count=?
                       WHERE id=?""",
                    (status, frame_count, tile_count or 0,
                     segment_count or 0, mission_id),
                )
            else:
                conn.execute(
                    "UPDATE missions SET status=? WHERE id=?",
                    (status, mission_id),
                )
            conn.commit()

    def list_missions(self) -> list[Mission]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM missions ORDER BY created_at DESC"
            ).fetchall()
        return [self._row_to_mission(r) for r in rows]
