"""Timeline recorder for one simulation run - what the 3D studio plays back.

Attach it before the swarm is built::

    rec = TraceRecorder(keep_frames=True)
    sim.tracer = rec
    build_swarm(sim, ...); sim.run(duration)
    data = rec.to_dict()               # JSON-friendly rows, layouts in SCHEMA
    fids, images = rec.frame_arrays()  # what the cameras saw (uint8 HxW), if keep_frames

The recorder only observes (see ``Simulator.trace``): a traced run produces exactly the metrics of an
untraced one. Times are simulated seconds rounded to 1 us. Message kinds and CPU tags are stored as
indices into ``kinds`` / ``tags`` to keep traces of long runs small.
"""

from __future__ import annotations

from typing import Any

import numpy as np

SCHEMA: dict[str, list[str]] = {
    "tx": ["t_queue", "t_start", "t_end", "t_arrive", "src", "dst", "kind", "bytes", "lost", "fid"],
    "bcast": ["t_queue", "t_start", "t_end", "t_arrive", "src", "kind", "bytes", "lost_by", "fid"],
    "cpu": ["t_book", "t_start", "t_end", "node", "macs", "tag", "fid"],
    "capture": ["t", "node", "fid", "label"],
    "infer": ["t", "node", "fid", "pred", "conf", "label"],
    "learn": ["t", "node", "fid", "pred", "conf", "label", "t_capture"],
    "reject": ["t", "node", "fid"],
    "event": ["t", "kind", "node", "by"],
    "role": ["t", "node", "leader"],
    "camera": ["t", "node", "on"],
    "log": ["t", "text"],
}
BULK = ("tx", "bcast", "cpu", "log")  # dropped first when a run is too long to keep everything


def _r(t: float) -> float:
    return round(float(t), 6)


def _fid(payload: Any) -> int:
    """Frame id carried by a message payload: a Frame, or a tuple whose first element is the id."""
    if hasattr(payload, "t_capture"):
        return int(payload.id)
    if isinstance(payload, tuple) and payload and isinstance(payload[0], int):
        return payload[0]
    return -1


class TraceRecorder:
    def __init__(self, keep_frames: bool = False, max_rows: int = 3_000_000, max_frame_bytes: int = 64 << 20):
        self.keep_frames = keep_frames
        self.max_rows = max_rows
        self.max_frame_bytes = max_frame_bytes
        self.rows: dict[str, list[list]] = {k: [] for k in SCHEMA}
        self.kinds: list[str] = []
        self.tags: list[str] = []
        self._kind_ix: dict[str, int] = {}
        self._tag_ix: dict[str, int] = {}
        self._cpu_by_node: dict[int, list[list]] = {}
        self._n = 0
        self.truncated = False
        self._frames: dict[int, np.ndarray] = {}
        self._frame_bytes = 0
        self.frames_truncated = False

    # ------------------------------------------------------------------ helpers
    def _code(self, name: str, table: list[str], index: dict[str, int]) -> int:
        i = index.get(name)
        if i is None:
            i = index[name] = len(table)
            table.append(name)
        return i

    def _add(self, key: str, row: list) -> list | None:
        if self._n >= self.max_rows and key in BULK:
            self.truncated = True
            return None
        self._n += 1
        self.rows[key].append(row)
        return row

    # ------------------------------------------------------------------ Simulator.tracer protocol
    def __call__(self, now: float, kind: str, data: tuple) -> None:
        handler = getattr(self, f"_on_{kind}", None)
        if handler is not None:
            handler(now, *data)

    def _on_tx(self, now, src, dst, msg, t_start, t_end, t_arrive, lost) -> None:
        self._add("tx", [_r(now), _r(t_start), _r(t_end), _r(t_arrive), src, dst,
                         self._code(msg.kind, self.kinds, self._kind_ix), int(msg.size), int(bool(lost)),
                         _fid(msg.payload)])

    def _on_bcast(self, now, src, msg, t_start, t_end, t_arrive, lost_by) -> None:
        self._add("bcast", [_r(now), _r(t_start), _r(t_end), _r(t_arrive), src,
                            self._code(msg.kind, self.kinds, self._kind_ix), int(msg.size), list(lost_by),
                            _fid(msg.payload)])

    def _on_cpu(self, now, node, t_start, t_end, macs, cb_name, fid) -> None:
        tag = "brain" if cb_name == "_brain_done" else cb_name.strip("_") or "work"
        row = self._add("cpu", [_r(now), _r(t_start), _r(t_end), node, int(macs),
                                self._code(tag, self.tags, self._tag_ix), int(fid)])
        if row is not None:
            self._cpu_by_node.setdefault(node, []).append(row)

    def _on_capture(self, now, node, frame) -> None:
        self._add("capture", [_r(now), node, int(frame.id), int(frame.label)])
        if self.keep_frames and not self.frames_truncated:
            img = np.asarray(frame.image, dtype=np.float32)
            if img.ndim == 3:
                img = img[0]  # brains see (C,H,W); the sensor view shows the first channel
            u8 = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
            if self._frame_bytes + u8.nbytes > self.max_frame_bytes:
                self.frames_truncated = True
            else:
                self._frames[int(frame.id)] = u8
                self._frame_bytes += u8.nbytes

    def _on_infer(self, now, node, fid, pred, conf, label) -> None:
        self._add("infer", [_r(now), node, int(fid), int(pred), round(float(conf), 4), int(label)])

    def _on_learn(self, now, node, fid, pred, conf, label, t_capture) -> None:
        self._add("learn", [_r(now), node, int(fid), int(pred), round(float(conf), 4),
                            -1 if label is None else int(label), _r(t_capture if t_capture is not None else now)])

    def _on_reject(self, now, node, fid) -> None:
        self._add("reject", [_r(now), node, int(fid)])

    def _on_event(self, now, kind, node, by) -> None:
        self._add("event", [_r(now), kind, node, -1 if by is None else by])
        if kind == "leader":
            self._add("role", [_r(now), node, 1])
        elif kind == "kill":
            self._add("role", [_r(now), node, 0])
            # work booked before the crash never completes (Node.kill bumps its epoch): cut it at the kill
            for row in reversed(self._cpu_by_node.get(node, [])):
                if row[2] <= now:
                    break
                row[2] = max(row[1], _r(now))

    def _on_step_down(self, now, node) -> None:
        self._add("role", [_r(now), node, 0])

    def _on_camera(self, now, node, on) -> None:
        self._add("camera", [_r(now), node, int(bool(on))])

    def _on_log(self, now, text) -> None:
        self._add("log", [_r(now), str(text)])

    # ------------------------------------------------------------------ output
    def to_dict(self) -> dict:
        out: dict[str, Any] = {"schema": SCHEMA, "kinds": list(self.kinds), "tags": list(self.tags)}
        for key, rows in self.rows.items():
            out[key] = [r for r in rows if r[2] > r[1]] if key == "cpu" else rows
        out["truncated"] = self.truncated
        out["frames"] = {"kept": len(self._frames), "truncated": self.frames_truncated}
        return out

    def frame_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """``(fids int64[N], images uint8[N,H,W])`` in capture order; empty arrays if no frames were kept."""
        if not self._frames:
            return np.zeros(0, dtype=np.int64), np.zeros((0, 1, 1), dtype=np.uint8)
        fids = np.fromiter(self._frames.keys(), dtype=np.int64, count=len(self._frames))
        return fids, np.stack(list(self._frames.values()))
