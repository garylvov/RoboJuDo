"""Rich rate/throughput + latency recorder for deploy post-mortem debugging.

Tracks, per named component (e.g. "camera", "robot_state", "policy_inference",
"command_send", "env_step"): instantaneous & average Hz (derived from
inter-event timestamps) and optional per-event latency. A bounded ring buffer of
recent events is flushed to disk (jsonl or csv) by a background writer thread so
the data survives a crash.

Enable via ``RecorderCfg(enabled=True)``; when disabled the recorder is a cheap
no-op so it is safe to leave ``rec.mark(...)`` / ``with rec.measure(...)`` calls
in hot deploy loops.
"""

import csv
import json
import logging
import os
import queue
import threading
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Literal, Optional

try:
    from robojudo.config import Config
except ModuleNotFoundError:  # allow `python robojudo/tools/recorder.py` from repo root
    import sys

    _repo_root = Path(__file__).resolve().parents[2]
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))
    from robojudo.config import Config

logger = logging.getLogger(__name__)


class RecorderCfg(Config):
    enabled: bool = False
    output_dir: str = "logs/recordings"
    format: Literal["jsonl", "csv"] = "jsonl"
    ring_buffer_size: int = 100000  # max events held in memory ring buffer
    flush_interval_s: float = 1.0
    components: list[str] = [
        "camera",
        "robot_state",
        "policy_inference",
        "command_send",
        "env_step",
    ]
    ewma_alpha: float = 0.1  # smoothing for ewma Hz/latency


# recent-latency window used for the bounded p95 estimate
_LATENCY_WINDOW = 512


class _ComponentStats:
    """Running per-component statistics. Not internally locked; the owning
    RateRecorder holds a single lock around all mutations."""

    def __init__(self, ewma_alpha: float):
        self.ewma_alpha = ewma_alpha
        self.count = 0
        self.last_ts: Optional[float] = None
        # running mean of inter-event dt (simple running mean)
        self.mean_dt = 0.0
        self.ewma_hz: Optional[float] = None
        # latency running stats (values stored in ms)
        self.latency_count = 0
        self.mean_latency_ms = 0.0
        self.ewma_latency_ms: Optional[float] = None
        self.recent_latency_ms: deque = deque(maxlen=_LATENCY_WINDOW)

    def update(self, t: float, latency_s: Optional[float]) -> tuple[Optional[float], Optional[float]]:
        """Update stats for a new event at epoch time ``t``.

        Returns ``(dt, hz)`` for this event (either may be None on the first
        event or when dt is non-positive).
        """
        dt = None
        hz = None
        if self.last_ts is not None:
            dt = t - self.last_ts
            if dt > 0:
                hz = 1.0 / dt
                # simple running mean of dt
                self.mean_dt += (dt - self.mean_dt) / max(1, self.count)
                if self.ewma_hz is None:
                    self.ewma_hz = hz
                else:
                    a = self.ewma_alpha
                    self.ewma_hz = a * hz + (1.0 - a) * self.ewma_hz
        self.last_ts = t
        self.count += 1

        if latency_s is not None:
            lat_ms = latency_s * 1000.0
            self.latency_count += 1
            self.mean_latency_ms += (lat_ms - self.mean_latency_ms) / self.latency_count
            if self.ewma_latency_ms is None:
                self.ewma_latency_ms = lat_ms
            else:
                a = self.ewma_alpha
                self.ewma_latency_ms = a * lat_ms + (1.0 - a) * self.ewma_latency_ms
            self.recent_latency_ms.append(lat_ms)
        return dt, hz

    def mean_hz(self) -> Optional[float]:
        if self.mean_dt > 0:
            return 1.0 / self.mean_dt
        return None

    def p95_latency_ms(self) -> Optional[float]:
        if not self.recent_latency_ms:
            return None
        ordered = sorted(self.recent_latency_ms)
        # nearest-rank p95 over the bounded recent window
        idx = int(round(0.95 * (len(ordered) - 1)))
        return ordered[idx]

    def snapshot(self) -> dict:
        return {
            "count": self.count,
            "mean_hz": self.mean_hz(),
            "ewma_hz": self.ewma_hz,
            "mean_latency_ms": (self.mean_latency_ms if self.latency_count else None),
            "p95_latency_ms": self.p95_latency_ms(),
            "last_ts": self.last_ts,
        }


class RateRecorder:
    """Rate/throughput + latency recorder with a background disk flusher.

    Public API::

        rec = RateRecorder(cfg, run_name="deploy0")
        rec.mark("robot_state")
        rec.mark("command_send", latency_s=0.002, meta={"joint": 3})
        with rec.measure("policy_inference"):
            action = policy(obs)
        rec.stats()
        rec.close()
    """

    _CSV_HEADER = ["t", "component", "dt", "hz", "latency_ms", "meta"]

    def __init__(self, cfg: RecorderCfg, run_name: Optional[str] = None):
        self.cfg = cfg
        self.enabled = bool(cfg.enabled)
        self._lock = threading.Lock()
        self._stats: dict[str, _ComponentStats] = {}
        self._ring: dict[str, deque] = {}
        self.rec_dir: Optional[Path] = None
        self.events_path: Optional[Path] = None

        if not self.enabled:
            return

        timestamp = time.strftime("%y%m%d-%H%M%S")
        name = f"rec_{timestamp}"
        if run_name:
            name += f"_{run_name}"
        self.rec_dir = Path(cfg.output_dir) / name
        os.makedirs(self.rec_dir, exist_ok=True)

        ext = "jsonl" if cfg.format == "jsonl" else "csv"
        self.events_path = self.rec_dir / f"events.{ext}"
        # line-buffered text file; writer thread flushes+fsyncs.
        self._fh = open(self.events_path, "w", newline="")
        self._csv_writer = None
        if cfg.format == "csv":
            self._csv_writer = csv.writer(self._fh)
            self._csv_writer.writerow(self._CSV_HEADER)
            self._fh.flush()
            os.fsync(self._fh.fileno())

        # pre-create stats/ring for configured components (others created lazily)
        for comp in cfg.components:
            self._stats[comp] = _ComponentStats(cfg.ewma_alpha)
            self._ring[comp] = deque(maxlen=cfg.ring_buffer_size)

        self._queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._writer_thread, daemon=True)
        self._thread.start()
        logger.info("RateRecorder writing to %s", self.events_path)

    # ------------------------------------------------------------------ #
    def _ensure_component(self, component: str) -> _ComponentStats:
        st = self._stats.get(component)
        if st is None:
            st = _ComponentStats(self.cfg.ewma_alpha)
            self._stats[component] = st
            self._ring[component] = deque(maxlen=self.cfg.ring_buffer_size)
        return st

    def mark(
        self,
        component: str,
        latency_s: Optional[float] = None,
        meta: Optional[dict] = None,
    ) -> None:
        """Record that ``component`` produced an event *now*."""
        if not self.enabled:
            return
        t = time.time()
        with self._lock:
            st = self._ensure_component(component)
            dt, hz = st.update(t, latency_s)
            record = {
                "t": t,
                "component": component,
                "dt": dt,
                "hz": hz,
                "latency_ms": (latency_s * 1000.0 if latency_s is not None else None),
                "meta": meta,
            }
            self._ring[component].append(record)
        # queue is thread-safe on its own; keep it outside the lock
        self._queue.put(record)

    @contextmanager
    def measure(self, component: str, meta: Optional[dict] = None):
        """Context manager that times the wrapped block and marks its latency.

        Usage::

            with rec.measure("policy_inference"):
                action = policy(obs)
        """
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.mark(component, latency_s=elapsed, meta=meta)

    def stats(self) -> dict:
        """Return a JSON-serializable per-component stats snapshot."""
        with self._lock:
            return {comp: st.snapshot() for comp, st in self._stats.items()}

    # ------------------------------------------------------------------ #
    def _write_record(self, record: dict) -> None:
        if self.cfg.format == "jsonl":
            self._fh.write(json.dumps(record, default=str) + "\n")
        else:
            meta = record.get("meta")
            meta_col = "" if meta is None else json.dumps(meta, default=str)
            self._csv_writer.writerow(
                [
                    record["t"],
                    record["component"],
                    "" if record["dt"] is None else record["dt"],
                    "" if record["hz"] is None else record["hz"],
                    "" if record["latency_ms"] is None else record["latency_ms"],
                    meta_col,
                ]
            )

    def _drain_queue(self) -> bool:
        wrote = False
        while True:
            try:
                record = self._queue.get_nowait()
            except queue.Empty:
                break
            self._write_record(record)
            self._queue.task_done()
            wrote = True
        return wrote

    def _writer_thread(self) -> None:
        while not self._stop_event.is_set():
            # block briefly for the first record to avoid a busy loop
            try:
                first = self._queue.get(timeout=self.cfg.flush_interval_s)
            except queue.Empty:
                continue
            self._write_record(first)
            self._queue.task_done()
            self._drain_queue()
            self._fh.flush()
            os.fsync(self._fh.fileno())
        # final drain after stop requested
        if self._drain_queue():
            self._fh.flush()
            os.fsync(self._fh.fileno())

    def close(self) -> None:
        """Stop the flush thread, write ``summary.json`` and close files."""
        if not self.enabled:
            return
        try:
            self._stop_event.set()
            self._thread.join()
        except Exception as e:  # pragma: no cover - best effort on shutdown
            logger.warning("RateRecorder writer join failed: %s", e)

        try:
            summary = {
                "cfg": self.cfg.to_dict(),
                "stats": self.stats(),
                "closed_at": time.time(),
            }
            summary_path = self.rec_dir / "summary.json"
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)
                f.flush()
                os.fsync(f.fileno())
        except Exception as e:  # pragma: no cover
            logger.warning("RateRecorder summary write failed: %s", e)

        try:
            self._fh.flush()
            os.fsync(self._fh.fileno())
            self._fh.close()
        except Exception as e:  # pragma: no cover
            logger.warning("RateRecorder file close failed: %s", e)
        self.enabled = False


if __name__ == "__main__":
    import tempfile

    logging.basicConfig(level=logging.INFO)

    out_dir = os.path.join(tempfile.gettempdir(), "robojudo_recorder_smoke")
    cfg = RecorderCfg(enabled=True, output_dir=out_dir, format="jsonl")
    rec = RateRecorder(cfg, run_name="smoke")

    for i in range(50):
        rec.mark("robot_state")
        # time a tiny block of "inference"
        with rec.measure("policy_inference"):
            time.sleep(0.002)
        # command send with a fake latency
        rec.mark("command_send", latency_s=0.0015, meta={"iter": i})
        # sleep to produce a realistic ~50-200 Hz outer rate
        time.sleep(0.005)

    stats = rec.stats()
    print("=== rec.stats() ===")
    print(json.dumps(stats, indent=2, default=str))

    rec.close()

    print("\nevents file:", rec.events_path)
    print("summary.json:", rec.rec_dir / "summary.json")

    print("\n=== first 3 event lines ===")
    with open(rec.events_path) as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            print(line.rstrip())

    print("\n=== summary.json ===")
    with open(rec.rec_dir / "summary.json") as f:
        print(f.read())
