"""
core/streaming.py — Live stream ingestion manager.

Accepts an RTSP/RTMP/file URL and runs two parallel pipelines:

  1. HLS transcoding  — ffmpeg writes .m3u8 + .ts segments to hls_dir/.
     The browser plays these via hls.js with ~5–10 s latency.

  2a. Frame extraction (no embedded telemetry) — ffmpeg writes a JPEG every
      FRAME_INTERVAL_S seconds to frames_dir/.  A background thread picks up
      new frames, calls on_frames(paths, lat, lon) with the fixed GPS anchor
      supplied at start, and periodically triggers on_analyse().

  2b. Segment extraction (embedded djmd telemetry detected) — ffmpeg muxes ALL
      tracks into SEGMENT_DURATION_S-long MP4 chunks written to segments_dir/.
      A background thread processes each completed segment with DJIProtobufParser
      + VideoSampler (both from core/video.py) to extract per-frame GPS, altitude,
      and gimbal angles from the djmd protobuf stream, then calls
      on_frames_telem([(jpeg_path, SRTFrame), ...]).

      Detection: ffprobe probes the source URL before start.  If a stream with
      codec_tag_string == "djmd" or handler_name containing "dji meta" is found,
      telemetry mode activates automatically.

Thread safety
-------------
All state is guarded by a single threading.Lock().  The stream manager is a
singleton instantiated once in main.py and shared across requests.
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
FRAME_INTERVAL_S   = 2.0    # JPEG extraction rate for the fixed-anchor path
SEGMENT_DURATION_S = 5      # MP4 segment length for the djmd telemetry path
ANALYSIS_EVERY_N   = 10     # trigger on_analyse() after this many new frames
HLS_SEGMENT_S      = 2      # HLS segment duration in seconds
HLS_LIST_SIZE      = 5      # HLS playlist window size
FFMPEG_LOGLEVEL    = "warning"
POLL_INTERVAL_S    = 0.5    # background thread polling interval
DJMD_PROBE_TIMEOUT = 12     # ffprobe timeout for djmd stream detection (seconds)


class StreamManager:
    """
    Lifecycle: idle → starting → running → stopping → idle
    A single instance is shared across the application.
    """

    def __init__(self) -> None:
        self._lock               = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._running            = False
        self._stopping           = False
        self._frame_count        = 0
        self._error: str | None  = None
        self._url: str           = ""
        self._hls_dir: Path | None      = None
        self._frames_dir: Path | None   = None
        self._segments_dir: Path | None = None
        self._telemetry_enabled  = False
        self._source_telem: list = []   # pre-parsed SRTFrames from source (djmd path)
        # Callbacks wired by main.py at startup
        self._on_frames:       Callable[[list[Path], float, float], None] | None = None
        self._on_frames_telem: Callable[[list[tuple]], None] | None = None
        self._on_analyse:      Callable[[], None] | None = None

    # ── public API ────────────────────────────────────────────────────────────

    def init(
        self,
        on_frames:       Callable[[list[Path], float, float], None],
        on_analyse:      Callable[[], None],
        on_frames_telem: Callable[[list[tuple]], None] | None = None,
    ) -> None:
        """
        Wire up callbacks from main.py.

        on_frames(paths, lat, lon)
            Fixed-anchor path: list of new JPEG paths + operator-supplied GPS.
            Used when no djmd stream is present in the source.

        on_frames_telem(pairs)
            Telemetry path: list of (jpeg_path, SRTFrame) tuples, one per
            sampled frame extracted from a completed MP4 segment.  SRTFrame
            carries real GPS, altitude, and gimbal angles from the djmd stream.

        on_analyse()
            Called every ANALYSIS_EVERY_N frames; triggers detection pipeline.
        """
        self._on_frames       = on_frames
        self._on_frames_telem = on_frames_telem
        self._on_analyse      = on_analyse

    def start(
        self,
        url:          str,
        lat:          float,
        lon:          float,
        hls_dir:      Path,
        frames_dir:   Path,
        segments_dir: Path | None = None,
    ) -> None:
        """
        Probe the source for a djmd telemetry track, then start ffmpeg and
        the appropriate background processing thread.
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

            # Pre-parse all djmd telemetry from the source now.
            # Segments will carry video only (ffmpeg cannot copy the djmd track
            # to a new mp4 — codec_id=NONE is rejected by the mp4 muxer).
            # Telemetry is matched to segment frames by absolute timestamp.
            try:
                from core.video import DJIProtobufParser
                self._source_telem = DJIProtobufParser().parse(Path(url))
                logger.info(
                    "Pre-parsed %d djmd telemetry frames from source",
                    len(self._source_telem),
                )
            except Exception as exc:
                logger.warning("Failed to pre-parse djmd telemetry: %s", exc)
                self._source_telem = []

            logger.info(
                "djmd stream found at index %d — telemetry-aware segment path active",
                djmd_idx,
            )
        else:
            self._source_telem = []
            logger.info(
                "No djmd stream in '%s' — using fixed GPS anchor (%.6f, %.6f)",
                url, lat, lon,
            )

        # ── ffmpeg command ────────────────────────────────────────────────────
        m3u8      = hls_dir / "stream.m3u8"
        frame_pat = frames_dir / "live_%05d.jpg"

        cmd = [
            "ffmpeg", "-y", "-loglevel", FFMPEG_LOGLEVEL,
        ]
        # -rtsp_transport is an RTSP-demuxer option; passing it before a local
        # file path causes ffmpeg to abort with "Option not found".
        if url.lower().startswith(("rtsp://", "rtsps://")):
            cmd += ["-rtsp_transport", "tcp"]
        cmd += [
            "-i", url,
            # Output 1: HLS for browser playback (always present)
            "-map", "0:v:0",
            "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
            "-g", str(HLS_SEGMENT_S * 30),
            "-sc_threshold", "0",
            "-f", "hls",
            "-hls_time",      str(HLS_SEGMENT_S),
            "-hls_list_size", str(HLS_LIST_SIZE),
            "-hls_flags",     "delete_segments+omit_endlist+append_list",
            str(m3u8),
        ]

        if djmd_idx is None:
            # Output 2a: JPEG frames at fixed rate, fixed GPS anchor
            cmd += [
                "-map", "0:v:0",
                "-vf", f"fps=1/{FRAME_INTERVAL_S}",
                "-q:v", "3",
                "-update", "0",
                str(frame_pat),
            ]
        else:
            # Output 2b: video-only MP4 segments.
            #
            # We do NOT use -map 0 here because the djmd data track has
            # codec_id=NONE which the mp4 muxer rejects ("Could not find tag
            # for codec none in stream").  Telemetry was pre-parsed from the
            # source above and is matched to frames by absolute timestamp.
            #
            # -reset_timestamps is NOT used so that segment frame timestamps
            # remain absolute (relative to source start), which lets
            # VideoSampler / SRTParser.interpolate match them against the
            # pre-parsed SRTFrame list by absolute timestamp_ms.
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
            raise RuntimeError(
                "ffmpeg not found. Install it: https://ffmpeg.org/download.html"
            )

        with self._lock:
            self._proc    = proc
            self._running = True

        threading.Thread(
            target=self._read_stderr, args=(proc,), daemon=True,
        ).start()

        if djmd_idx is None:
            t = threading.Thread(
                target=self._process_frames,
                args=(frames_dir, lat, lon),
                daemon=True,
            )
        else:
            t = threading.Thread(
                target=self._process_segments,
                args=(seg_dir, frames_dir, self._source_telem),
                daemon=True,
            )
        t.start()
        with self._lock:
            self._thread = t

        logger.info("Stream started: %s (telemetry=%s)", url, djmd_idx is not None)

    def stop(self) -> None:
        """Terminate ffmpeg and signal the processing thread to exit."""
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
        """
        Probe the source URL with ffprobe.  Return the stream index of the
        djmd data track, or None if not present or probe fails.
        """
        cmd = ["ffprobe", "-v", "error"]
        if url.lower().startswith(("rtsp://", "rtsps://")):
            cmd += ["-rtsp_transport", "tcp"]
        cmd += ["-show_streams", "-of", "json", url]
        try:
            res = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=DJMD_PROBE_TIMEOUT,
            )
            info = json.loads(res.stdout)
        except Exception as exc:
            logger.debug("djmd probe failed for '%s': %s", url, exc)
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
            if any(k in low for k in ("error", "invalid", "no such file", "failed")):
                logger.warning("ffmpeg ERROR: %s", text)
                with self._lock:
                    self._error = text[:200]
            else:
                logger.warning("ffmpeg: %s", text)

    # ── Path A: fixed-anchor JPEG processing ───────────────────────────────────

    def _process_frames(self, frames_dir: Path, lat: float, lon: float) -> None:
        """
        Poll frames_dir for new JPEGs written by ffmpeg's fps filter.
        Call on_frames(paths, lat, lon) and trigger on_analyse() periodically.
        All frames share the fixed GPS anchor (lat, lon) supplied at start().
        """
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
                logger.info(
                    "Stream: %d new frame(s) — total %d",
                    len(new), len(processed) + len(new),
                )
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

    # ── Path B: djmd telemetry segment processing ──────────────────────────────

    def _process_segments(self, segments_dir: Path, frames_dir: Path, source_telem: list) -> None:
        """
        Poll segments_dir for completed video-only MP4 segments.
        source_telem is the full list of SRTFrames pre-parsed from the djmd
        track of the source file; it is passed to VideoSampler so that each
        segment's extracted frames get real per-frame GPS and gimbal data
        matched by absolute timestamp.
        """
        try:
            self._run_segment_loop(segments_dir, frames_dir, source_telem)
        except Exception as exc:
            logger.error(
                "Segment processor thread crashed: %s", exc, exc_info=True
            )
        finally:
            logger.info("Segment processor thread exiting.")

    def _run_segment_loop(self, segments_dir: Path, frames_dir: Path, source_telem: list) -> None:
        """Inner loop — separated so the outer method can catch all exceptions."""
        from core.video import VideoSampler

        video_sampler = VideoSampler()
        processed: set[str] = set()
        since_analysis = 0

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

            # A segment is complete when either:
            #   (a) a newer sibling exists (ffmpeg has moved on), or
            #   (b) it is the sole/last file AND its mtime is older than
            #       SEGMENT_DURATION_S + 2 s (ffmpeg has finished writing it)
            now = time.time()
            complete = [
                s for s in segs
                if s.name not in processed
                and (
                    s is not segs[-1]                               # has a newer sibling
                    or now - s.stat().st_mtime > SEGMENT_DURATION_S + 2  # or sealed by age
                )
            ]

            for seg in complete:
                self._process_one_segment(seg, frames_dir, source_telem, video_sampler)
                processed.add(seg.name)
                since_analysis += 1
                if since_analysis >= max(1, ANALYSIS_EVERY_N // SEGMENT_DURATION_S) \
                        and self._on_analyse:
                    since_analysis = 0
                    try:
                        self._on_analyse()
                    except Exception as exc:
                        logger.error("on_analyse error: %s", exc)
                try:
                    seg.unlink()
                except Exception:
                    pass

            if stopping:
                break
            time.sleep(POLL_INTERVAL_S)

    def _process_one_segment(
        self,
        seg_path:     Path,
        frames_dir:   Path,
        source_telem: list,   # pre-parsed list[SRTFrame] from source file
        video_sampler,
    ) -> None:
        """
        Extract video frames from one MP4 segment and pair them with telemetry
        from the pre-parsed source SRTFrame list using absolute timestamps.

        Segments carry video only (djmd was parsed from the source directly).
        VideoSampler extracts frames whose PTS timestamps are absolute (relative
        to source start), so SRTParser.interpolate matches them correctly
        against the full source_telem list.
        """
        if not source_telem:
            logger.warning("Segment %s: no source telemetry available — skipped", seg_path.name)
            return

        try:
            frame_pairs = video_sampler.sample_file(
                seg_path, srt_frames=source_telem, out_dir=frames_dir,
            )
        except Exception as exc:
            logger.warning("Segment %s VideoSampler error: %s", seg_path.name, exc)
            return

        if not frame_pairs:
            logger.debug("Segment %s: no sampled frames", seg_path.name)
            return

        n  = len(frame_pairs)
        f0 = frame_pairs[0][1]
        logger.info(
            "Segment %s → %d frame(s) | GPS (%.5f, %.5f) | alt %.0fm",
            seg_path.name, n, f0.lat, f0.lon, f0.alt_m,
        )

        if self._on_frames_telem:
            try:
                self._on_frames_telem(frame_pairs)
            except Exception as exc:
                logger.error("on_frames_telem error: %s", exc)

        with self._lock:
            self._frame_count += n