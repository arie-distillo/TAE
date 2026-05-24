"""
core/streaming.py — Live stream ingestion manager.

Accepts an RTSP/RTMP/file URL and runs two parallel pipelines:

  1. HLS transcoding  — ffmpeg writes .m3u8 + .ts segments to hls_dir/.
     The browser plays these via hls.js with ~5–10 s latency.

  2. Frame extraction — ffmpeg writes a JPEG every FRAME_INTERVAL_S seconds
     to frames_dir/. A background thread picks up new frames, indexes them
     with CLIP, and periodically triggers auto-analysis against the mission
     definition so the map updates in near-real-time.

GPS without telemetry
---------------------
Live streams don't carry a DJI SRT sidecar, so per-frame GPS is not
available without a MAVLink link (Phase E+).  For Phase D, the operator
provides a fixed anchor position (lat, lon) at stream start; all indexed
frames are tagged with that position.  When MAVLink is integrated later,
this module can be extended to interpolate per-frame GPS from the
telemetry log.

Thread safety
-------------
All state is guarded by a single threading.Lock().  The stream manager is
a singleton instantiated once in main.py and shared across requests.
"""

import logging
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable

logger = logging.getLogger("TAE.Stream")

# ── tunables ──────────────────────────────────────────────────────────────────
FRAME_INTERVAL_S   = 2.0    # extract 1 JPEG every N seconds
ANALYSIS_EVERY_N   = 10     # run auto-analysis after this many new frames
HLS_SEGMENT_S      = 2      # HLS segment duration in seconds
HLS_LIST_SIZE      = 5      # number of segments kept in the playlist
FFMPEG_LOGLEVEL    = "warning"
POLL_INTERVAL_S    = 0.5    # how often the thread checks for new frames


class StreamManager:
    """
    Lifecycle: idle → starting → running → stopping → idle
    A single instance is shared across the application.
    """

    def __init__(self) -> None:
        self._lock        = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._running     = False
        self._stopping    = False
        self._frame_count = 0
        self._error: str | None = None
        self._url: str   = ""
        self._hls_dir: Path | None = None
        self._frames_dir: Path | None = None
        # Callbacks injected by main.py at startup
        self._on_frames:  Callable[[list[Path], float, float], None] | None = None
        self._on_analyse: Callable[[], None] | None = None

    # ── public API ────────────────────────────────────────────────────────────

    def init(
        self,
        on_frames:  Callable[[list[Path], float, float], None],
        on_analyse: Callable[[], None],
    ) -> None:
        """
        Wire up callbacks from main.py.

        on_frames(paths, lat, lon)  — called with a batch of new JPEG paths;
                                      main.py indexes them into LanceDB and
                                      builds their meta entries.
        on_analyse()                — called every ANALYSIS_EVERY_N frames;
                                      main.py runs _execute_analysis() against
                                      mission.definition and rebuilds the map.
        """
        self._on_frames  = on_frames
        self._on_analyse = on_analyse

    def start(
        self,
        url:         str,
        lat:         float,
        lon:         float,
        hls_dir:     Path,
        frames_dir:  Path,
        track_query: str   = "",
        track_conf:  float = 0.15,
    ) -> None:
        """
        Start the ffmpeg subprocess and background frame-processor thread.
        Raises RuntimeError if already running.
        """
        with self._lock:
            if self._running:
                raise RuntimeError("Stream already running — stop it first.")
            self._url         = url
            self._hls_dir     = hls_dir
            self._frames_dir  = frames_dir
            self._frame_count = 0
            self._error       = None
            self._stopping    = False
            self._track_query = track_query
            self._track_conf  = track_conf

        hls_dir.mkdir(parents=True, exist_ok=True)
        frames_dir.mkdir(parents=True, exist_ok=True)

        if track_query:
            try:
                from core.tracker import init_stream_tracker
                init_stream_tracker(track_query)
                logger.info("Stream tracker init: %s", track_query)
            except Exception as e:
                logger.warning("Could not init stream tracker: %s", e)

        m3u8 = hls_dir / "stream.m3u8"
        frame_pat = frames_dir / "live_%05d.jpg"

        cmd = [
            "ffmpeg", "-y", "-loglevel", FFMPEG_LOGLEVEL,
            # Input — accept any ffmpeg-readable URL (rtsp/rtmp/file/http)
            "-i", url,
            # ── Output 1: HLS ──────────────────────────────────────────────
            "-map", "0:v:0",
            "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
            "-g", str(HLS_SEGMENT_S * 30),   # keyframe every segment
            "-sc_threshold", "0",
            "-f", "hls",
            "-hls_time", str(HLS_SEGMENT_S),
            "-hls_list_size", str(HLS_LIST_SIZE),
            "-hls_flags", "delete_segments+omit_endlist+append_list",
            str(m3u8),
            # ── Output 2: JPEG frames for CLIP/VLM ────────────────────────
            "-map", "0:v:0",
            "-vf", f"fps=1/{FRAME_INTERVAL_S}",
            "-q:v", "3",           # JPEG quality 1–31 (lower = better)
            "-update", "0",        # write sequentially numbered files
            str(frame_pat),
        ]

        logger.info("Starting ffmpeg: %s", " ".join(cmd))
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "ffmpeg not found. Install it: https://ffmpeg.org/download.html"
            )

        with self._lock:
            self._proc    = proc
            self._running = True

        # Stderr reader to surface ffmpeg errors without blocking
        threading.Thread(
            target=self._read_stderr, args=(proc,), daemon=True
        ).start()

        # Frame processor
        t = threading.Thread(
            target=self._process_frames,
            args=(frames_dir, lat, lon),
            daemon=True,
        )
        t.start()
        with self._lock:
            self._thread = t

        logger.info("Stream started: %s", url)

    def stop(self) -> None:
        """Terminate the ffmpeg process and wait for the processor thread."""
        with self._lock:
            if not self._running:
                return
            self._stopping = True
            proc = self._proc

        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

        with self._lock:
            self._running  = False
            self._stopping = False
            self._proc     = None

        logger.info("Stream stopped.")
        try:
            from core.tracker import reset_stream_tracker
            reset_stream_tracker()
        except Exception:
            pass

    @property
    def running(self) -> bool:
        with self._lock:
            return self._running

    def status(self) -> dict:
        with self._lock:
            return {
                "running":     self._running,
                "url":         self._url,
                "frame_count": self._frame_count,
                "error":       self._error,
                "hls_ready":   (
                    (self._hls_dir / "stream.m3u8").exists()
                    if self._hls_dir else False
                ),
            }

    def hls_playlist_path(self) -> Path | None:
        with self._lock:
            return (self._hls_dir / "stream.m3u8") if self._hls_dir else None

    # ── internal ─────────────────────────────────────────────────────────────

    def _read_stderr(self, proc: subprocess.Popen) -> None:
        """Surface ffmpeg errors to the TAE logger without blocking."""
        for line in proc.stderr:
            text = line.decode(errors="ignore").strip()
            if text:
                logger.debug("ffmpeg: %s", text)
                # Surface genuine errors (not warnings) to _error
                low = text.lower()
                if any(k in low for k in ("error", "invalid", "no such file")):
                    with self._lock:
                        self._error = text[:200]

    def _process_frames(self, frames_dir: Path, lat: float, lon: float) -> None:
        """
        Background thread: poll frames_dir for new JPEGs, call on_frames(),
        and trigger on_analyse() every ANALYSIS_EVERY_N frames.
        """
        processed: set[str] = set()
        since_analysis = 0

        while True:
            with self._lock:
                if not self._running:
                    break
                stopping = self._stopping

            # Collect new frames (sorted by mtime so they arrive in order)
            try:
                new = sorted(
                    [
                        p for p in frames_dir.glob("live_*.jpg")
                        if p.name not in processed
                    ],
                    key=lambda p: p.stat().st_mtime,
                )
            except Exception:
                new = []

            if new:
                logger.info(
                    "Stream: %d new frame(s) — total %d",
                    len(new), len(processed) + len(new),
                )
                if self._on_frames:
                    try:
                        self._on_frames(new, lat, lon)
                    except Exception as e:
                        logger.error("on_frames error: %s", e)

                # Per-frame SORT tracking (streaming path)
                if self._track_query:
                    for fp in new:
                        try:
                            import cv2 as _cv
                            from core.tracker import track_one_frame
                            img = _cv.imread(str(fp))
                            if img is None:
                                continue
                            fh, fw = img.shape[:2]
                            fp_map = {
                                "nw": (lat, lon), "ne": (lat, lon),
                                "se": (lat, lon), "sw": (lat, lon),
                            }
                            track_one_frame(
                                frame_path   = fp,
                                frame_name   = fp.name,
                                frame_idx    = len(processed),
                                timestamp_ms = int(time.time() * 1000),
                                footprint    = fp_map,
                                frame_w      = fw,
                                frame_h      = fh,
                                tracks_file  = (
                                    self._hls_dir.parent / "maps" / "tracks.json"
                                ),
                                confidence   = self._track_conf,
                            )
                        except Exception as e:
                            logger.warning("Per-frame tracking error: %s", e)

                for p in new:
                    processed.add(p.name)

                with self._lock:
                    self._frame_count = len(processed)

                since_analysis += len(new)
                if since_analysis >= ANALYSIS_EVERY_N and self._on_analyse:
                    since_analysis = 0
                    try:
                        self._on_analyse()
                    except Exception as e:
                        logger.error("on_analyse error: %s", e)

            if stopping:
                break

            time.sleep(POLL_INTERVAL_S)

        logger.info("Frame processor thread exiting.")
