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

Schema additions vs Phase 1:
  - last_accessed (INTEGER, Unix ms) — updated on every activation; used to
    sort the mission dropdown by recency.
  - status gains a new value "archived" — soft-delete that hides a mission
    from the dropdown without destroying its data.  Existing operational
    values "created" | "ingesting" | "ready" are unchanged.
  - global_state table — single key/value row tracking the last active
    mission_id so the server resumes the right mission after a restart.
"""

import json
import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

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
    upload_path:      str         # absolute path to the frames directory
    lancedb_path:     str         # absolute path to the LanceDB directory
    segments_db_path: str         # absolute path to segments.db
    created_at:       str         # ISO-8601 UTC
    status:           str         # "created"|"ingesting"|"ready"|"archived"
    frame_count:      int  = 0
    tile_count:       int  = 0
    segment_count:    int  = 0
    scene_context:    str  = ""   # optional operator hint for anomaly queries
    definition:       str  = ""   # natural-language mission goal; auto-applied on ingestion
    last_accessed:    Optional[int] = None  # Unix ms; None until first activation

    def allows(self, intent: str) -> bool:
        return intent in self.allowed_intents

    @property
    def short_id(self) -> str:
        return self.id[:8]

    @property
    def is_archived(self) -> bool:
        return self.status == "archived"


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
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
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
                    scene_context    TEXT    NOT NULL DEFAULT '',
                    definition       TEXT    NOT NULL DEFAULT '',
                    last_accessed    INTEGER            -- Unix ms, NULL until first activation
                )
            """)

            # Migrate: add last_accessed to tables created before this column existed
            try:
                conn.execute("ALTER TABLE missions ADD COLUMN last_accessed INTEGER")
                logger.info("Migrated missions table: added last_accessed column")
            except sqlite3.OperationalError:
                pass  # column already present — normal on all runs after first migration

            try:
                conn.execute("ALTER TABLE missions ADD COLUMN definition TEXT NOT NULL DEFAULT ''")
                logger.info("Migrated missions table: added definition column")
            except sqlite3.OperationalError:
                pass

            # global_state: persists last_active_mission_id across server restarts
            conn.execute("""
                CREATE TABLE IF NOT EXISTS global_state (
                    key   TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            conn.execute(
                "INSERT OR IGNORE INTO global_state VALUES "
                "('last_active_mission_id', NULL)"
            )
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
            definition       = row["definition"] or "",
            last_accessed    = row["last_accessed"],
        )

    # ── Create ────────────────────────────────────────────────────────────────

    def create(
        self,
        name:             str,
        upload_path:      str | Path,
        lancedb_path:     str | Path,
        segments_db_path: str | Path,
        allowed_intents:  list[str] | None = None,
        mission_id:       str | None = None,
    ) -> Mission:
        """Explicitly create a new mission (used by the multi-mission UI flow)."""
        mission_id = mission_id or uuid.uuid4().hex
        intents    = allowed_intents or ALLOWED_INTENTS_DEFAULT
        now        = datetime.utcnow().isoformat()

        with self._connect() as conn:
            conn.execute(
                """INSERT INTO missions
                       (id, name, allowed_intents, upload_path, lancedb_path,
                        segments_db_path, created_at, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 'created')""",
                (
                    mission_id, name.strip(),
                    json.dumps(intents),
                    str(upload_path),
                    str(lancedb_path),
                    str(segments_db_path),
                    now,
                ),
            )
            conn.commit()

        mission = Mission(
            id               = mission_id,
            name             = name.strip(),
            allowed_intents  = intents,
            upload_path      = str(upload_path),
            lancedb_path     = str(lancedb_path),
            segments_db_path = str(segments_db_path),
            created_at       = now,
            status           = "created",
        )
        logger.info(
            f"Mission created | '{mission.name}' ({mission.short_id}) | "
            f"intents: {intents}"
        )
        return mission

    # ── Read ──────────────────────────────────────────────────────────────────

    def get(self, mission_id: str) -> Optional[Mission]:
        """Return a single mission by ID, or None if not found."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM missions WHERE id=?", (mission_id,)
            ).fetchone()
        return self._row_to_mission(row) if row else None

    def list_active(self) -> list[Mission]:
        """All non-archived missions, most recently accessed first."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM missions
                   WHERE status != 'archived'
                   ORDER BY last_accessed DESC, created_at DESC"""
            ).fetchall()
        return [self._row_to_mission(r) for r in rows]

    def list_archived(self) -> list[Mission]:
        """All archived missions, most recently accessed first."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM missions
                   WHERE status = 'archived'
                   ORDER BY last_accessed DESC, created_at DESC"""
            ).fetchall()
        return [self._row_to_mission(r) for r in rows]

    def list_missions(self) -> list[Mission]:
        """All missions regardless of status (backward compat)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM missions ORDER BY created_at DESC"
            ).fetchall()
        return [self._row_to_mission(r) for r in rows]

    # ── Update ────────────────────────────────────────────────────────────────

    def update(
        self,
        mission_id:      str,
        name:            str | None = None,
        allowed_intents: list[str] | None = None,
        scene_context:   str | None = None,
        definition:      str | None = None,
    ) -> None:
        """Update editable mission fields (called from the settings drawer)."""
        sets, vals = [], []
        if name is not None:
            sets.append("name=?");            vals.append(name.strip())
        if allowed_intents is not None:
            sets.append("allowed_intents=?"); vals.append(json.dumps(allowed_intents))
        if scene_context is not None:
            sets.append("scene_context=?");   vals.append(scene_context)
        if definition is not None:
            sets.append("definition=?");      vals.append(definition.strip())
        if not sets:
            return
        vals.append(mission_id)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE missions SET {', '.join(sets)} WHERE id=?", vals
            )
            conn.commit()

    def update_counts(
        self,
        mission_id:    str,
        status:        str,
        frame_count:   int | None = None,
        tile_count:    int | None = None,
        segment_count: int | None = None,
    ) -> None:
        """Update operational status and optional counts after ingestion."""
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

    def touch(self, mission_id: str) -> None:
        """Record that this mission was just activated (updates last_accessed)."""
        ts = int(time.time() * 1000)
        with self._connect() as conn:
            conn.execute(
                "UPDATE missions SET last_accessed=? WHERE id=?",
                (ts, mission_id),
            )
            conn.commit()

    # ── Archive / Delete ──────────────────────────────────────────────────────

    def archive(self, mission_id: str) -> None:
        """Soft-delete: hide from the active list but keep all data on disk."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE missions SET status='archived' WHERE id=?",
                (mission_id,),
            )
            conn.commit()
        logger.info(f"Mission archived ({mission_id[:8]})")

    def delete(self, mission_id: str) -> None:
        """
        Remove the DB record.  Caller is responsible for wiping DATA_DIR/{id}/
        before calling this — main.py's /missions/{id} DELETE route does so.
        """
        with self._connect() as conn:
            conn.execute("DELETE FROM missions WHERE id=?", (mission_id,))
            conn.commit()
        logger.info(f"Mission deleted from DB ({mission_id[:8]})")

    # ── Global state ──────────────────────────────────────────────────────────

    def get_last_active(self) -> Optional[str]:
        """Return the mission_id that was active when the server last shut down."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM global_state "
                "WHERE key='last_active_mission_id'"
            ).fetchone()
        return row["value"] if (row and row["value"]) else None

    def set_last_active(self, mission_id: str) -> None:
        """Persist the active mission_id so it survives a server restart."""
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO global_state VALUES "
                "('last_active_mission_id', ?)",
                (mission_id,),
            )
            conn.commit()

    # ── Phase-1 compat ────────────────────────────────────────────────────────

    def get_or_create_active(
        self,
        upload_path:      str | Path,
        lancedb_path:     str | Path,
        segments_db_path: str | Path,
        allowed_intents:  list[str] | None = None,
    ) -> Mission:
        """
        Phase-1 startup helper: resume the most recent non-archived mission
        whose upload_path matches, or create a new one.

        Still called by _restore_state() in main.py.  Once the full
        multi-mission UI flow is wired up this method can be retired.
        """
        upload_str = str(upload_path)
        with self._connect() as conn:
            row = conn.execute(
                """SELECT * FROM missions
                   WHERE upload_path=? AND status != 'archived'
                   ORDER BY created_at DESC LIMIT 1""",
                (upload_str,),
            ).fetchone()

        if row:
            mission = self._row_to_mission(row)
            logger.info(
                f"Mission resumed | '{mission.name}' ({mission.short_id}) | "
                f"intents: {mission.allowed_intents} | status: {mission.status}"
            )
            return mission

        # First run for this upload_path — create a new mission
        name = f"Mission {datetime.now():%Y-%m-%d %H:%M}"
        return self.create(
            name             = name,
            upload_path      = upload_path,
            lancedb_path     = lancedb_path,
            segments_db_path = segments_db_path,
            allowed_intents  = allowed_intents,
        )