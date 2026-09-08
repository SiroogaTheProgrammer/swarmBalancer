"""Bounded, inference-only mesh scheduler. Peers cannot arm hardware or send code.

Membership is advisory, not distributed consensus. Each input owner schedules its
own idempotent inference, avoiding a single mandatory command node. Results are
not actuator commands. A trusted plugin must return promptly; Python cannot kill
a stuck C/Python thread. Such a worker keeps its capacity occupied until it exits.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import logging
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

from .config import NodeConfig, Peer, positive
from .protocol import (HEADER, JOB_PREFIX, LOAD, VERSION, Kind, ProtocolError, authorized_peer,
                       read_packet, tls_context, validate_local_identity)

_LOG = logging.getLogger(__name__)
_BUSY, _EXPIRED, _FAILED = 1, 2, 3
MAX_SESSION_S = 3600.0  # Bound session age; re-handshakes recheck certificate validity/pins.


class JobError(RuntimeError):
    """No timely authorized inference is available; the local app must fail safely."""


@dataclass(frozen=True)
class InferenceResult:
    payload: bytes
    useful: bool = True

    def __post_init__(self):
        if not isinstance(self.payload, bytes) or type(self.useful) is not bool:
            raise ValueError("inference result must contain bytes and a boolean useful flag")


@dataclass
class _Work:
    key: tuple[str, bytes]
    payload: bytes
    digest: bytes
    deadline: float
    future: asyncio.Future


class _Session:
    def __init__(self, node: SwarmRuntime, peer: Peer, reader, writer):
        self.node, self.peer, self.reader, self.writer = node, peer, reader, writer
        self.write_lock = asyncio.Lock()
        self.pending: dict[bytes, asyncio.Future] = {}
        self.reservations = 0  # Includes jobs selected before their asynchronous send begins.
        self.replying: set[bytes] = set()
        self.reply_tasks: set[asyncio.Task] = set()
        self.queued = self.capacity = 0  # No advertised capacity -> do not send work yet.
        self.estimated_ms = node.config.estimated_ms
        self.last_seen = time.monotonic()

    async def send(self, kind: Kind, payload: bytes):
        async def write():
            async with self.write_lock:
                if self.writer.is_closing():
                    raise ConnectionError("peer disconnected")
                self.writer.write(HEADER.pack(kind, len(payload)) + payload)
                await self.writer.drain()
                self.node.counters["application_bytes_sent"] += HEADER.size + len(payload)

        try:
            await asyncio.wait_for(write(), self.node.config.peer_timeout_s)
        except (OSError, asyncio.TimeoutError):
            self.writer.close()
            raise

    async def request(self, job_id: bytes, payload: bytes, deadline: float) -> InferenceResult:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise JobError("deadline expired")
        future = asyncio.get_running_loop().create_future()
        self.pending[job_id] = future
        try:
            ttl_ms = max(1, int(remaining * 1000))
            await self.send(Kind.JOB, JOB_PREFIX.pack(job_id, ttl_ms) + payload)
            return await asyncio.wait_for(asyncio.shield(future), max(0, deadline - time.monotonic()))
        finally:
            self.pending.pop(job_id, None)
            if not future.done():
                future.cancel()
            elif not future.cancelled():
                future.exception()  # Drain disconnect errors even if the sender was cancelled.

    def disconnect(self):
        self.writer.close()
        for future in self.pending.values():
            if not future.done():
                future.set_exception(JobError("peer disconnected; inference outcome unknown"))


class SwarmRuntime:
    """Use ``async with SwarmRuntime(config, handler) as node`` then ``await node.submit(bytes)``.

    ``handler`` is synchronous trusted LOCAL code. Async/network work never runs
    inside it; it must be pure/idempotent because a lost reply can trigger a retry
    on a different worker. There is no exactly-once or safety-command guarantee.
    """

    def __init__(self, config: NodeConfig, handler: Callable[[bytes], InferenceResult]):
        if not callable(handler) or inspect.iscoroutinefunction(handler):
            raise TypeError("handler must be a synchronous local callable")
        self.config, self.handler = config, handler
        self._sessions: dict[str, _Session] = {}
        self._sessions_starting: set[str] = set()
        self._inbound_connections = 0
        self._tasks: set[asyncio.Task] = set()
        self._queue: asyncio.Queue[_Work] = asyncio.Queue(config.queue_limit)
        self._work: dict[tuple[str, bytes], _Work] = {}
        self._cache: OrderedDict[tuple[str, bytes], tuple[float, bytes, InferenceResult]] = OrderedDict()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="swarm-inference")
        self._active = False
        self._submissions = 0
        self._closing = False
        self._started = False
        self._server = None
        self._changed = asyncio.Event()
        self._estimated_ms = config.estimated_ms
        self.counters = {key: 0 for key in ("submitted", "completed", "retried", "rejected", "local_jobs",
                                           "failed_jobs", "suppressed_results", "application_bytes_sent",
                                           "application_bytes_received", "peer_connections", "peer_disconnects")}

    @property
    def port(self) -> int:
        if not self._server:
            raise RuntimeError("runtime has not started")
        return self._server.sockets[0].getsockname()[1]

    @property
    def online_peers(self) -> tuple[str, ...]:
        return tuple(sorted(self._sessions))

    @property
    def leader_id(self) -> str:
        """Informational preference only; NOT a quorum/lease or permission to control hardware."""
        return min((self.config.node_id, *self._sessions))

    def status(self) -> dict:
        return {"node_id": self.config.node_id, "workload_id": self.config.workload_id,
                "online_peers": self.online_peers, "preferred_leader": self.leader_id,
                "queued": len(self._work), "capacity": self.config.queue_limit,
                "estimated_ms": self._estimated_ms, "counters": dict(self.counters)}

    def _spawn(self, coroutine):
        task = asyncio.create_task(coroutine)
        self._tasks.add(task)

        def finished(done):
            self._tasks.discard(done)
            if not done.cancelled() and done.exception():
                _LOG.error("Runtime background task failed: %s", type(done.exception()).__name__)

        task.add_done_callback(finished)
        return task

    async def start(self):
        if self._started or self._closing:
            raise RuntimeError("runtime instances may only be started once")
        server_tls = tls_context(self.config, server=True)
        self._client_tls = tls_context(self.config, server=False)
        validate_local_identity(self.config, server_tls, self._client_tls)
        self._server = await asyncio.start_server(
            self._incoming, self.config.host, self.config.port, ssl=server_tls,
            ssl_handshake_timeout=self.config.connect_timeout_s,
            limit=self.config.max_payload + 1024, backlog=32)
        self._started = True
        self._spawn(self._worker())
        for peer in self.config.peers:
            # Exactly one dialer per pair, independent of which endpoint started first.
            if self.config.node_id < peer.node_id:
                self._spawn(self._connect(peer))
        return self

    def _incoming(self, reader, writer):
        if self._closing or self._inbound_connections >= len(self.config.peers):
            writer.close()
            return
        # Reserve synchronously, before the coroutine runs: a burst of completed
        # TLS handshakes must not create an unbounded batch of app-level tasks.
        self._inbound_connections += 1
        task = self._spawn(self._serve(reader, writer, None))

        def released(_):
            self._inbound_connections -= 1

        task.add_done_callback(released)

    async def _connect(self, peer: Peer):
        delay = 0.1
        while not self._closing:
            try:
                reader, writer = await asyncio.wait_for(asyncio.open_connection(
                    peer.host, peer.port, ssl=self._client_tls, server_hostname=peer.node_id,
                    ssl_handshake_timeout=self.config.connect_timeout_s,
                    limit=self.config.max_payload + 1024), self.config.connect_timeout_s)
                delay = 0.1
                await self._serve(reader, writer, peer)
            except (OSError, asyncio.TimeoutError, ConnectionError):
                pass
            if not self._closing:
                await asyncio.sleep(delay)
                delay = min(delay * 2, 5.0)

    async def _serve(self, reader, writer, expected):
        session = None
        beat = None
        reserved_id = None
        try:
            peer = authorized_peer(writer, self.config, expected)
            if expected is None and peer.node_id > self.config.node_id:
                raise ProtocolError("the lower node ID must initiate this connection")
            if peer.node_id in self._sessions or peer.node_id in self._sessions_starting:
                raise ProtocolError("duplicate peer session")
            self._sessions_starting.add(peer.node_id)
            reserved_id = peer.node_id
            session = _Session(self, peer, reader, writer)
            # IDs come from pinned certificates; hello only checks protocol/workload compatibility.
            hello = bytes([VERSION]) + self.config.workload_id.encode("ascii")
            await session.send(Kind.HELLO, hello)
            kind, body = await asyncio.wait_for(read_packet(reader, self.config.max_payload), self.config.connect_timeout_s)
            if kind != Kind.HELLO or body != hello:
                raise ProtocolError("peer protocol/workload mismatch")
            self._sessions[peer.node_id] = session
            self._sessions_starting.discard(peer.node_id)
            self.counters["peer_connections"] += 1
            beat = self._spawn(self._heartbeat(session))
            self._changed.set()
            session_started = time.monotonic()
            while not self._closing:
                lifetime = MAX_SESSION_S - (time.monotonic() - session_started)
                if lifetime <= 0:
                    break
                kind, body = await asyncio.wait_for(read_packet(reader, self.config.max_payload),
                                                    min(self.config.peer_timeout_s, lifetime))
                session.last_seen = time.monotonic()
                self.counters["application_bytes_received"] += HEADER.size + len(body)
                await self._message(session, kind, body)
        except (OSError, ConnectionError, asyncio.TimeoutError, asyncio.IncompleteReadError, ValueError):
            pass  # Never log untrusted packet bodies or certificate/key contents.
        finally:
            if reserved_id:
                self._sessions_starting.discard(reserved_id)
            if session is not None:
                for task in list(session.reply_tasks):
                    task.cancel()
                if self._sessions.get(session.peer.node_id) is session:
                    self._sessions.pop(session.peer.node_id)
                    self.counters["peer_disconnects"] += 1
                session.disconnect()
            if beat:
                beat.cancel()
            writer.close()
            self._changed.set()
            try:
                await asyncio.wait_for(writer.wait_closed(), 0.5)
            except (OSError, ConnectionError, asyncio.TimeoutError):
                pass

    async def _heartbeat(self, session: _Session):
        try:
            while not self._closing:
                estimate = max(1, min(300000000, int(self._estimated_ms * 1000)))
                await session.send(Kind.LOAD, LOAD.pack(len(self._work), self.config.queue_limit, estimate))
                await asyncio.sleep(self.config.heartbeat_s)
        except (OSError, ConnectionError, asyncio.TimeoutError):
            session.disconnect()

    async def _message(self, session: _Session, kind: Kind, body: bytes):
        if kind == Kind.LOAD:
            queued, capacity, estimate = LOAD.unpack(body)
            if not 1 <= capacity <= 64 or queued > capacity or not 1 <= estimate <= 300000000:
                raise ProtocolError("invalid load advertisement")
            session.queued, session.capacity = queued, capacity
            session.estimated_ms = estimate / 1000
            self._changed.set()
        elif kind == Kind.JOB:
            if len(body) < JOB_PREFIX.size:
                raise ProtocolError("truncated job")
            job_id, ttl = JOB_PREFIX.unpack_from(body)
            if not 1 <= ttl <= int(self.config.job_timeout_s * 1000):
                raise ProtocolError("job TTL exceeds local policy")
            if len(session.replying) >= self.config.queue_limit and job_id not in session.replying:
                await session.send(Kind.ERROR, job_id + bytes([_BUSY]))
                return
            try:
                future = self._accept(session.peer.node_id, job_id, body[JOB_PREFIX.size:], time.monotonic() + ttl / 1000)
            except JobError:
                await session.send(Kind.ERROR, job_id + bytes([_BUSY]))
                return
            if job_id not in session.replying:
                session.replying.add(job_id)
                reply = self._spawn(self._reply(session, job_id, future, ttl / 1000))
                session.reply_tasks.add(reply)
                reply.add_done_callback(session.reply_tasks.discard)
        elif kind in (Kind.RESULT, Kind.ERROR):
            if len(body) < 17:
                raise ProtocolError("truncated result")
            job_id, flag = body[:16], body[16]
            if kind == Kind.RESULT and (flag not in (0, 1) or (flag == 0 and len(body) != 17)):
                raise ProtocolError("invalid result flags/payload")
            if kind == Kind.ERROR and flag not in (_BUSY, _EXPIRED, _FAILED):
                raise ProtocolError("unknown job error")
            future = session.pending.get(job_id)
            if future is not None and not future.done():
                if kind == Kind.ERROR:
                    if flag == _BUSY:
                        session.queued = session.capacity
                    future.set_exception(JobError({1: "peer busy", 2: "peer deadline expired", 3: "peer inference failed"}[flag]))
                else:
                    session.queued = max(0, session.queued - 1)
                    future.set_result(InferenceResult(body[17:], bool(flag)))
        else:
            raise ProtocolError("unexpected message after handshake")

    async def _reply(self, session, job_id, future, ttl):
        try:
            try:
                result = await asyncio.wait_for(asyncio.shield(future), ttl)
                await session.send(Kind.RESULT, job_id + bytes([result.useful]) + result.payload)
            except asyncio.TimeoutError:
                await session.send(Kind.ERROR, job_id + bytes([_EXPIRED]))
            except JobError:
                await session.send(Kind.ERROR, job_id + bytes([_FAILED]))
        except (OSError, ConnectionError, asyncio.TimeoutError):
            session.disconnect()
        finally:
            session.replying.discard(job_id)

    def _accept(self, owner: str, job_id: bytes, payload: bytes, deadline: float) -> asyncio.Future:
        key = owner, job_id
        digest = hashlib.sha256(payload).digest()
        now = time.monotonic()
        while self._cache and next(iter(self._cache.values()))[0] <= now:
            self._cache.popitem(last=False)
        if key in self._work:
            work = self._work[key]
            if work.digest != digest:
                raise ProtocolError("reused job ID with different input")
            return work.future
        if key in self._cache:
            _, previous_digest, result = self._cache[key]
            if digest != previous_digest:
                raise ProtocolError("reused job ID with different input")
            future = asyncio.get_running_loop().create_future()
            future.set_result(result)
            return future
        if self._closing or len(self._work) >= self.config.queue_limit:
            self.counters["rejected"] += 1
            raise JobError("worker queue full")
        future = asyncio.get_running_loop().create_future()
        # Owners may go offline or time out; always consume unobserved error futures.
        future.add_done_callback(lambda f: None if f.cancelled() else f.exception())
        work = _Work(key, payload, digest, deadline, future)
        self._work[key] = work
        self._queue.put_nowait(work)
        return future

    async def _worker(self):
        while not self._closing:
            work = await self._queue.get()
            try:
                if work.deadline <= time.monotonic():
                    raise JobError("job expired before execution")
                self._active = True
                start = time.monotonic()
                self.counters["local_jobs"] += 1
                result = await asyncio.get_running_loop().run_in_executor(self._executor, self.handler, work.payload)
                self._estimated_ms = max(0.001, 0.8 * self._estimated_ms + 0.2 * (time.monotonic() - start) * 1000)
                if not isinstance(result, InferenceResult) or len(result.payload) > self.config.max_payload:
                    raise JobError("plugin returned an invalid or oversized result")
                if time.monotonic() >= work.deadline:
                    raise JobError("inference completed after deadline")
                if not result.useful:
                    result = InferenceResult(b"", False)
                    self.counters["suppressed_results"] += 1
                if self.config.cache_entries:
                    self._cache[work.key] = (time.monotonic() + self.config.job_timeout_s * 2, work.digest, result)
                    while len(self._cache) > self.config.cache_entries:
                        self._cache.popitem(last=False)
                if not work.future.done():
                    work.future.set_result(result)
            except asyncio.CancelledError:
                if not work.future.done():
                    work.future.set_exception(JobError("runtime stopped"))
                raise
            except Exception:
                self.counters["failed_jobs"] += 1
                if not work.future.done():
                    work.future.set_exception(JobError("local inference failed or expired"))
            finally:
                self._active = False
                self._work.pop(work.key, None)
                self._queue.task_done()
                self._changed.set()

    def _select(self, excluded: set[str]):
        candidates = []
        if self.config.node_id not in excluded and len(self._work) < self.config.queue_limit:
            candidates.append(((len(self._work) + 1) * self._estimated_ms, self.config.node_id, None))
        for node_id, session in self._sessions.items():
            pending = max(session.queued, session.reservations)
            if (node_id not in excluded and pending < session.capacity
                    and time.monotonic() - session.last_seen < self.config.peer_timeout_s
                    and not session.writer.is_closing()):
                candidates.append(((pending + 1) * session.estimated_ms, node_id, session))
        return min(candidates, key=lambda c: (c[0], c[1])) if candidates else None

    async def submit(self, payload: bytes, *, timeout_s: float | None = None) -> InferenceResult:
        if not self._started or self._closing:
            raise JobError("runtime is not running")
        if not isinstance(payload, bytes) or len(payload) > self.config.max_payload:
            raise ValueError("submit requires bytes within max_payload; preprocess large observations locally")
        timeout = positive(self.config.job_timeout_s if timeout_s is None else timeout_s, "timeout_s", self.config.job_timeout_s)
        if self._submissions >= self.config.max_submissions:
            self.counters["rejected"] += 1
            raise JobError("submission limit reached")
        job_id = uuid.uuid4().bytes
        self._submissions += 1
        self.counters["submitted"] += 1
        deadline = time.monotonic() + timeout
        excluded: set[str] = set()
        try:
            attempts = min(self.config.retries + 1, len(self.config.peers) + 1)
            for attempt in range(attempts):
                choice = self._select(excluded)
                if choice is None or self._closing:
                    raise JobError("no authorized worker has available capacity")
                _, node_id, session = choice
                excluded.add(node_id)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                # Leave a bounded portion of the caller's budget for retries elsewhere.
                attempt_deadline = time.monotonic() + remaining / (attempts - attempt)
                try:
                    if session is None:
                        future = self._accept(self.config.node_id, job_id, payload, attempt_deadline)
                        result = await asyncio.wait_for(asyncio.shield(future), max(0, attempt_deadline - time.monotonic()))
                    else:
                        session.reservations += 1
                        try:
                            result = await asyncio.wait_for(session.request(job_id, payload, attempt_deadline),
                                                            max(0, attempt_deadline - time.monotonic()))
                        finally:
                            session.reservations -= 1
                    if time.monotonic() >= deadline:
                        raise JobError("late result discarded")
                    self.counters["completed"] += 1
                    return result
                except (JobError, OSError, ConnectionError, asyncio.TimeoutError):
                    if attempt + 1 < attempts:
                        self.counters["retried"] += 1
            raise JobError("no worker returned a result before its deadline")
        finally:
            self._submissions -= 1

    async def wait_for_peers(self, count: int, timeout_s: float = 10.0):
        if not 0 <= count <= len(self.config.peers):
            raise ValueError("requested peer count exceeds configured peers")

        async def ready():
            while sum(s.capacity > 0 for s in self._sessions.values()) < count:
                self._changed.clear()
                await self._changed.wait()

        await asyncio.wait_for(ready(), timeout_s)

    async def close(self):
        if self._closing:
            return
        self._closing = True
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        for session in list(self._sessions.values()):
            session.disconnect()
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for work in self._work.values():
            if not work.future.done():
                work.future.set_exception(JobError("runtime stopped"))
        self._work.clear()
        self._cache.clear()
        self._executor.shutdown(wait=False, cancel_futures=True)

    async def __aenter__(self):
        return await self.start()

    async def __aexit__(self, *exc):
        await self.close()