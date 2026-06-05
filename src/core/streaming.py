"""
core/streaming.py — Live stream ingestion manager.

Two processing paths depending on whether the source carries an embedded
djmd telemetry stream:

Path A — no embedded telemetry (fixed GPS anchor)
    ffmpeg writes JPEG frames at FRAME_INTERVAL_S to frames_dir/.
    Background thread batches new frames → on_frames(paths, lat, lon).

Path B — djmd telemetry detected (per-frame real-time)
    ffmpeg segments the source into SEGMENT_DURATION_S MP4 chunks.
    For each completed segment VideoSampler extracts frames.
    Each frame is handed individually to on_frame_telem(jpeg_path, srt_frame)
    in strict temporal order so the caller can:
      - CLIP-index the frame immediately
      - run detection on only that frame's tiles
      - update a persistent cross-frame tracker
      - rebuild the map as confirmed detections accumulate

    This is true near-real-time processing: index → detect → track → map,
    one frame at a time, rather than batch-after-all-frames.

Thread safety
─────────────
All mutable state is guarded by a single Lock.  The stream manager is a
singleton instantiated once in main.py.
"""

import json
import logging
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable

logger = logging.getLogger("TAE.Stream")

# ── tunables ──────────────────────────────────────────────────────────────────
FRAME_INTERVAL_S   = 2.0    # JPEG extraction rate (Path A only)
SEGMENT_DURATION_S = 5      # MP4 segment length in seconds
ANALYSIS_EVERY_N   = 5      # call on_analyse() every N segments (Path A)
HLS_SEGMENT_S      = 2
HLS_LIST_SIZE      = 5
FFMPEG_LOGLEVEL    = "warning"
POLL_INTERVAL_S    = 0.5
DJMD_PROBE_TIMEOUT = 12


class StreamManager:

    def __init__(self) -> None:
        self._lock               = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._running            = False
        self._stopping           = False
        self._frame_count        = 0
        self._timeline_ms        = 0     # absolute offset of next segment
        self._error: str | None  = None
        self._url: str           = ""
        self._hls_dir: Path | None      = None
        self._frames_dir: Path | None   = None
        self._segments_dir: Path | None = None
        self._telemetry_enabled  = False
        self._source_telem: list = []
        # Callbacks
        self._on_frames:      Callable | None = None   # Path A batch callback
        self._on_frame_telem: Callable | None = None   # Path B per-frame callback
        self._on_analyse:     Callable | None = None

    # ── public API ────────────────────────────────────────────────────────────

    def init(
        self,
        on_frames:      Callable[[list[Path], float, float], None],
        on_analyse:     Callable[[], None],
        on_frame_telem: Callable[[Path, object], None] | None = None,
    ) -> None:
        """
        Wire up callbacks.

        on_frames(paths, lat, lon)
            Path A (no djmd): batch of new JPEG paths + fixed GPS anchor.

        on_frame_telem(jpeg_path, srt_frame)
            Path B (djmd detected): one JPEG + its SRTFrame telemetry.
            Called in strict temporal order, one frame at a time.
            The caller is responsible for indexing, detecting, tracking and
            updating the map — all per frame.

        on_analyse()
            Path A only: periodic trigger for auto-analysis.
        """
        self._on_frames      = on_frames
        self._on_frame_telem = on_frame_telem
        self._on_analyse     = on_analyse

    def start(
        self,
        url:          str,
        lat:          float,
        lon:          float,
        hls_dir:      Path,
        frames_dir:   Path,
        segments_dir: Path | None = None,
    ) -> None:
        with self._lock:
            if self._running:
                raise RuntimeError("Stream already running — stop it first.")
            self._url         = url
            self._hls_dir     = hls_dir
            self._frames_dir  = frames_dir
            self._frame_count = 0
            self._timeline_ms = 0
            self._error       = None
            self._stopping    = False

        hls_dir.mkdir(parents=True, exist_ok=True)
        frames_dir.mkdir(parents=True, exist_ok=True)

        # ── Probe for embedded djmd telemetry ─────────────────────────────────
        djmd_idx = self._detect_djmd(url)
        with self._lock:
            self._telemetry_enabled = djmd_idx is not None

        if djmd_idx is not None:
            seg_dir = segments_dir or (hls_dir.parent / "live_segments")
            seg_dir.mkdir(parents=True, exist_ok=True)
            with self._lock:
                self._segments_dir = seg_dir

            # Pre-parse all djmd telemetry from the source.
            # ffmpeg cannot copy the djmd track to mp4 segments (codec_id=NONE
            # is rejected by the mp4 muxer), so we parse the source directly
            # and supply the full SRTFrame list to VideoSampler per segment.
            try:
                from core.video import DJIProtobufParser
                self._source_telem = DJIProtobufParser().parse(Path(url))
                logger.info(
                    "Pre-parsed %d djmd telemetry frames from source",
                    len(self._source_telem),
                )
            except Exception as exc:
                logger.warning("djmd pre-parse failed: %s", exc)
                self._source_telem = []

            logger.info(
                "djmd stream at index %d — per-frame real-time path active",
                djmd_idx,
            )
        else:
            self._source_telem = []
            logger.info(
                "No djmd stream in '%s' — fixed GPS anchor (%.6f, %.6f)",
                url, lat, lon,
            )

        # ── ffmpeg command ────────────────────────────────────────────────────
        m3u8      = hls_dir / "stream.m3u8"
        frame_pat = frames_dir / "live_%05d.jpg"

        cmd = ["ffmpeg", "-y", "-loglevel", FFMPEG_LOGLEVEL]
        if url.lower().startswith(("rtsp://", "rtsps://")):
            cmd += ["-rtsp_transport", "tcp"]
        cmd += [
            "-i", url,
            # HLS output — always present for browser playback
            "-map", "0:v:0",
            "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
            "-g", str(HLS_SEGMENT_S * 30), "-sc_threshold", "0",
            "-f", "hls",
            "-hls_time",      str(HLS_SEGMENT_S),
            "-hls_list_size", str(HLS_LIST_SIZE),
            "-hls_flags",     "delete_segments+omit_endlist+append_list",
            str(m3u8),
        ]

        if djmd_idx is None:
            # Path A: JPEG frames at fixed interval
            cmd += [
                "-map", "0:v:0",
                "-vf", f"fps=1/{FRAME_INTERVAL_S}",
                "-q:v", "3", "-update", "0",
                str(frame_pat),
            ]
        else:
            # Path B: video-only MP4 segments (djmd parsed separately above)
            cmd += [
                "-map", "0:v:0",
                "-c:v", "copy",
                "-f", "segment",
                "-segment_time",   str(SEGMENT_DURATION_S),
                "-segment_format", "mp4",
                str(seg_dir / "seg_%05d.mp4"),
            ]

        logger.info("Starting ffmpeg: %s", " ".join(cmd))
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            raise RuntimeError("ffmpeg not found.")

        with self._lock:
            self._proc    = proc
            self._running = True

        threading.Thread(target=self._read_stderr, args=(proc,), daemon=True).start()

        if djmd_idx is None:
            t = threading.Thread(
                target=self._process_frames, args=(frames_dir, lat, lon), daemon=True,
            )
        else:
            t = threading.Thread(
                target=self._process_segments, args=(seg_dir, frames_dir), daemon=True,
            )
        t.start()
        logger.info("Stream started: %s (telemetry=%s)", url, djmd_idx is not None)

    def stop(self) -> None:
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

    @property
    def running(self) -> bool:
        with self._lock:
            return self._running

    def status(self) -> dict:
        with self._lock:
            return {
                "running":           self._running,
                "url":               self._url,
                "frame_count":       self._frame_count,
                "error":             self._error,
                "telemetry_enabled": self._telemetry_enabled,
                "hls_ready": (
                    (self._hls_dir / "stream.m3u8").exists()
                    if self._hls_dir else False
                ),
            }

    def hls_playlist_path(self) -> Path | None:
        with self._lock:
            return (self._hls_dir / "stream.m3u8") if self._hls_dir else None

    # ── djmd detection ─────────────────────────────────────────────────────────

    def _detect_djmd(self, url: str) -> int | None:
        cmd = ["ffprobe", "-v", "error"]
        if url.lower().startswith(("rtsp://", "rtsps://")):
            cmd += ["-rtsp_transport", "tcp"]
        cmd += ["-show_streams", "-of", "json", url]
        try:
            res  = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=DJMD_PROBE_TIMEOUT)
            info = json.loads(res.stdout)
        except Exception as exc:
            logger.debug("djmd probe failed: %s", exc)
            return None
        for s in info.get("streams", []):
            tag     = s.get("codec_tag_string", "").lower()
            handler = s.get("tags", {}).get("handler_name", "").lower()
            if tag == "djmd" or "dji meta" in handler:
                return s["index"]
        return None

    # ── stderr reader ──────────────────────────────────────────────────────────

    def _read_stderr(self, proc: subprocess.Popen) -> None:
        for line in proc.stderr:
            text = line.decode(errors="ignore").strip()
            if not text:
                continue
            low = text.lower()
            if any(k in low for k in ("error", "invalid", "failed", "no such file")):
                logger.warning("ffmpeg ERROR: %s", text)
                with self._lock:
                    self._error = text[:200]
            else:
                logger.warning("ffmpeg: %s", text)

    # ── Path A: fixed-anchor JPEG processing ───────────────────────────────────

    def _process_frames(self, frames_dir: Path, lat: float, lon: float) -> None:
        processed: set[str] = set()
        since_analysis = 0
        while True:
            with self._lock:
                if not self._running:
                    break
                stopping = self._stopping
            try:
                new = sorted(
                    [p for p in frames_dir.glob("live_*.jpg")
                     if p.name not in processed],
                    key=lambda p: p.stat().st_mtime,
                )
            except Exception:
                new = []
            if new:
                if self._on_frames:
                    try:
                        self._on_frames(new, lat, lon)
                    except Exception as exc:
                        logger.error("on_frames error: %s", exc)
                for p in new:
                    processed.add(p.name)
                with self._lock:
                    self._frame_count = len(processed)
                since_analysis += len(new)
                if since_analysis >= ANALYSIS_EVERY_N and self._on_analyse:
                    since_analysis = 0
                    try:
                        self._on_analyse()
                    except Exception as exc:
                        logger.error("on_analyse error: %s", exc)
            if stopping:
                break
            time.sleep(POLL_INTERVAL_S)
        logger.info("Frame processor thread exiting.")

    # ── Path B: per-frame real-time processing ─────────────────────────────────

    def _process_segments(self, segments_dir: Path, frames_dir: Path) -> None:
        try:
            self._run_segment_loop(segments_dir, frames_dir)
        except Exception as exc:
            logger.error("Segment processor crashed: %s", exc, exc_info=True)
        finally:
            logger.info("Segment processor thread exiting.")

    def _run_segment_loop(self, segments_dir: Path, frames_dir: Path) -> None:
        from core.video import VideoSampler

        video_sampler = VideoSampler()
        processed: set[str] = set()

        logger.info("Segment loop started — watching %s", segments_dir)

        while True:
            with self._lock:
                if not self._running:
                    break
                stopping = self._stopping

            segs = sorted(segments_dir.glob("seg_*.mp4"))

            if segs:
                logger.info(
                    "Segment poll: %d file(s) in dir, %d already processed",
                    len(segs), len(processed),
                )

            # A segment is complete when a newer sibling exists, or when it is
            # old enough that ffmpeg has certainly finished writing it.
            now = time.time()
            complete = [
                s for s in segs
                if s.name not in processed
                and (
                    s is not segs[-1]
                    or now - s.stat().st_mtime > SEGMENT_DURATION_S + 2
                )
            ]

            for seg in complete:
                self._process_one_segment(seg, frames_dir, video_sampler)
                processed.add(seg.name)
                try:
                    seg.unlink()
                except Exception:
                    pass

            if stopping:
                break
            time.sleep(POLL_INTERVAL_S)

    # helper to get the true duration of a segment file in ms (frame_count / fps).
    @staticmethod
    def _segment_duration_ms(seg_path: Path) -> int:
        """True duration of a segment file in ms (frame_count / fps)."""
        import cv2
        cap = cv2.VideoCapture(str(seg_path))
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()
        return int(n / fps * 1000) if fps > 0 else 0
    

    def _process_one_segment(
        self,
        seg_path:     Path,
        frames_dir:   Path,
        video_sampler,
    ) -> None:
        """
        Extract frames from one MP4 segment and process each individually.

        VideoSampler returns all frames for the segment at once (ffmpeg
        constraint) but we iterate through them one by one in temporal order,
        calling on_frame_telem(jpeg_path, srt_frame) for each.

        This gives the caller (main.py) the ability to:
          1. CLIP-index each frame immediately
          2. register its timestamp for the tracker
          3. run detection on that frame's tiles only
          4. update the persistent cross-frame tracker
          5. rebuild the map as confirmed detections accumulate

        Temporal order is guaranteed because VideoSampler sorts frames by PTS
        and source_telem is sorted by timestamp_ms from the full-video parse.
        """
        if not self._source_telem:
            logger.warning("Segment %s: no source telemetry — skipped", seg_path.name)
            return

        # Each segment occupies a contiguous slice of the source timeline.
        # ffmpeg's segment muxer cuts on keyframes, so segment durations are
        # NOT uniform (e.g. 25s then 16s) — accumulate the real per-segment
        # duration rather than assuming SEGMENT_DURATION_S.
        base_ms = self._timeline_ms
        try:
            frame_pairs = video_sampler.sample_file(
                seg_path, srt_frames=self._source_telem, out_dir=frames_dir,
                base_ms=base_ms,
            )
        except Exception as exc:
            logger.warning("Segment %s VideoSampler error: %s", seg_path.name, exc)
            return

        # Advance the absolute timeline by this segment's TRUE duration so the
        # next segment's frames resolve telemetry at the correct video time.
        self._timeline_ms = base_ms + self._segment_duration_ms(seg_path)

        if not frame_pairs:
            logger.debug("Segment %s: no sampled frames", seg_path.name)
            return

        n  = len(frame_pairs)
        f0 = frame_pairs[0][1]
        logger.info(
            "Segment %s → %d frame(s) | GPS (%.5f, %.5f) | alt %.0fm "
            "— processing frame by frame",
            seg_path.name, n, f0.lat, f0.lon, f0.alt_m,
        )

        # ── per-frame sequential processing ───────────────────────────────────
        for jpeg_path, srt_frame in frame_pairs:
            if self._on_frame_telem:
                try:
                    self._on_frame_telem(jpeg_path, srt_frame)
                except Exception as exc:
                    logger.error(
                        "on_frame_telem error for %s: %s", jpeg_path.name, exc,
                    )
            with self._lock:
                self._frame_count += 1
