"""POSIX local JSONL persistence with durable commits and process-safe CAS.

Only newline-terminated, hash-checked records are authoritative. Checkpoints
are disposable hints; replay always reads the complete committed journal.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from types import TracebackType
from typing import Any, BinaryIO

from .hitl import _VALIDATORS, AgentPrompt, InvalidPromptResponse, PromptKind, PromptStatus
from .memory import (
    MemoryForgottenError,
    MemoryItem,
    MemoryKind,
    MemoryScope,
    MemoryStatus,
    memory_fingerprint,
)

JOURNAL_VERSION = 1
_NAMESPACE = re.compile(r"[A-Za-z0-9_-]+\Z")
_TOKEN = re.compile(r"[0-9a-f]{64}\Z")


class PersistenceError(Exception):
    """Local persistence could not safely commit or read an operation."""


class CorruptJournalError(PersistenceError):
    """A committed record failed validation; execution must fail closed."""


class StoreBusy(PersistenceError):
    """Another process holds the requested nonblocking journal lock."""


def _canonical(value: Any) -> bytes:
    def check(item: Any) -> None:
        if item is None or type(item) in (bool, int, str):
            return
        if type(item) is float:
            if not math.isfinite(item):
                raise ValueError("JSON numbers must be finite")
            return
        if isinstance(item, list):
            for child in item:
                check(child)
            return
        if isinstance(item, dict):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise ValueError("JSON object keys must be strings")
                check(child)
            return
        raise ValueError(f"unsupported JSON value {type(item).__name__}")

    check(value)
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant {value}")


def _parse(raw: bytes) -> Any:
    return json.loads(raw, object_pairs_hook=_pairs, parse_constant=_invalid_constant)


def _copy(value: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = _parse(_canonical(value))
    return result


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _private_directory(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise PersistenceError("journal directory is not a directory")
        return
    _private_directory(path.parent)
    try:
        path.mkdir(mode=0o700)
    except FileExistsError:
        if not path.is_dir():
            raise PersistenceError("journal directory is not a directory") from None
    _sync_directory(path.parent)


class FileJournal:
    """Exclusive-lock journal. Use only inside its context manager.

    The journal inode is stable: appends and tail repair never replace it, so
    ``flock`` also serializes independently constructed instances/processes.
    """

    def __init__(
        self, root: Path | str, namespace: str, key: str, *, blocking: bool = True
    ) -> None:
        if not _NAMESPACE.fullmatch(namespace):
            raise ValueError("journal namespace must be a plain directory name")
        if not isinstance(key, str):
            raise ValueError("journal key must be a string")
        self.root = Path(root)
        self.namespace = namespace
        self.blocking = blocking
        self._set_token(hashlib.sha256(key.encode("utf-8")).hexdigest())
        self._file: BinaryIO | None = None
        self._records: list[dict[str, Any]] = []

    def _set_token(self, token: str) -> None:
        self.path = self.root / self.namespace / f"{token}.jsonl"
        self.checkpoint_path = self.path.with_suffix(".checkpoint.json")

    @classmethod
    def _from_token(cls, root: Path, namespace: str, token: str) -> FileJournal:
        if not _TOKEN.fullmatch(token):
            raise CorruptJournalError("unexpected journal filename")
        journal = cls(root, namespace, "")
        journal._set_token(token)
        return journal

    @property
    def records(self) -> list[dict[str, Any]]:
        self._require_open()
        return [_copy(record) for record in self._records]

    @property
    def seq(self) -> int:
        return len(self._records)

    @property
    def digest(self) -> str:
        return self._records[-1]["digest"] if self._records else ""

    def _require_open(self) -> BinaryIO:
        if self._file is None:
            raise PersistenceError("journal must be used inside its context manager")
        return self._file

    def __enter__(self) -> FileJournal:
        if self._file is not None:
            raise PersistenceError("journal is already open")
        try:
            import fcntl
        except ImportError as exc:
            raise PersistenceError("FileJournal requires POSIX flock support") from exc
        try:
            _private_directory(self.root)
            directory = self.path.parent
            _private_directory(directory)
            if directory.is_symlink() or not directory.is_dir():
                raise PersistenceError("journal namespace must be a real directory")
            _sync_directory(self.root)
            descriptor = os.open(
                self.path, os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600
            )
            self._file = os.fdopen(descriptor, "r+b")
            operation = fcntl.LOCK_EX | (0 if self.blocking else fcntl.LOCK_NB)
            try:
                fcntl.flock(self._file.fileno(), operation)
            except BlockingIOError as exc:
                raise StoreBusy("journal is locked by another worker") from exc
            _sync_directory(directory)
            self._load()
            return self
        except OSError as exc:
            self._close()
            raise PersistenceError("could not open journal") from exc
        except BaseException:
            self._close()
            raise

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._close()

    def _close(self) -> None:
        if self._file is not None:
            file = self._file
            self._file = None
            file.close()

    def _load(self) -> None:
        file = self._require_open()
        file.seek(0)
        raw = file.read()
        committed_end = raw.rfind(b"\n") + 1
        records: list[dict[str, Any]] = []
        previous = ""
        for index, line in enumerate(raw[:committed_end].split(b"\n")[:-1], start=1):
            try:
                record = _parse(line)
                if not isinstance(record, dict) or set(record) != {
                    "version",
                    "seq",
                    "kind",
                    "data",
                    "prev_hash",
                    "digest",
                }:
                    raise ValueError("invalid record fields")
                if type(record["version"]) is not int or record["version"] != JOURNAL_VERSION:
                    raise ValueError("unsupported journal version")
                if type(record["seq"]) is not int or record["seq"] != index:
                    raise ValueError("non-contiguous journal sequence")
                if not isinstance(record["kind"], str) or not record["kind"]:
                    raise ValueError("invalid record kind")
                if not isinstance(record["data"], dict):
                    raise ValueError("record data must be an object")
                if record["prev_hash"] != previous:
                    raise ValueError("broken journal hash chain")
                content = {key: value for key, value in record.items() if key != "digest"}
                digest = hashlib.sha256(_canonical(content)).hexdigest()
                if record["digest"] != digest:
                    raise ValueError("journal digest mismatch")
                previous = digest
                records.append(record)
            except (TypeError, ValueError, UnicodeError) as exc:
                raise CorruptJournalError(f"invalid committed journal record {index}") from exc
        # A final line without its commit newline may be arbitrarily torn,
        # including in the middle of a UTF-8 sequence. Never try to decode it.
        if committed_end != len(raw):
            file.truncate(committed_end)
            file.flush()
            os.fsync(file.fileno())
        self._records = records

    def append(self, kind: str, data: dict[str, Any]) -> dict[str, Any]:
        file = self._require_open()
        try:
            if not isinstance(kind, str) or not kind or not isinstance(data, dict):
                raise ValueError("record requires a kind and an object")
            content = {
                "version": JOURNAL_VERSION,
                "seq": self.seq + 1,
                "kind": kind,
                "data": _copy(data),
                "prev_hash": self.digest,
            }
            record = {**content, "digest": hashlib.sha256(_canonical(content)).hexdigest()}
            encoded = _canonical(record) + b"\n"
        except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
            raise PersistenceError("record is not valid JSON") from exc
        try:
            file.seek(0, os.SEEK_END)
            file.write(encoded)
            file.flush()
            os.fsync(file.fileno())
        except OSError as exc:
            # The result is uncertain after an I/O failure; callers must reopen
            # and replay instead of issuing another append on stale state.
            try:
                self._close()
            except OSError:
                # Closing can repeat the failed flush; preserve the original
                # identifiable commit failure after disposing the old handle.
                pass
            raise PersistenceError("journal commit failed; reopen before retrying") from exc
        self._records.append(record)
        return _copy(record)

    def write_checkpoint(self, data: dict[str, Any]) -> None:
        self._require_open()
        temporary: str | None = None
        try:
            if not isinstance(data, dict):
                raise ValueError("checkpoint data must be an object")
            encoded = _canonical(
                {"version": JOURNAL_VERSION, "seq": self.seq, "digest": self.digest, "data": data}
            )
            descriptor, temporary = tempfile.mkstemp(
                prefix=f".{self.path.stem}.", dir=self.path.parent
            )
            with os.fdopen(descriptor, "wb") as file:
                file.write(encoded)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, self.checkpoint_path)
            temporary = None
            _sync_directory(self.path.parent)
        except (OSError, TypeError, ValueError, UnicodeError, RecursionError) as exc:
            raise PersistenceError("checkpoint commit failed") from exc
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)

    def read_checkpoint(self) -> dict[str, Any] | None:
        self._require_open()
        try:
            descriptor = os.open(self.checkpoint_path, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(descriptor, "rb") as file:
                checkpoint = _parse(file.read())
            if (
                not isinstance(checkpoint, dict)
                or set(checkpoint) != {"version", "seq", "digest", "data"}
                or type(checkpoint["version"]) is not int
                or checkpoint["version"] != JOURNAL_VERSION
                or type(checkpoint["seq"]) is not int
                or checkpoint["seq"] != self.seq
                or checkpoint["digest"] != self.digest
                or not isinstance(checkpoint["data"], dict)
            ):
                return None
            return _copy(checkpoint["data"])
        except (OSError, TypeError, ValueError, UnicodeError, RecursionError):
            return None


def _fields(data: dict[str, Any], names: set[str]) -> None:
    if set(data) != names:
        raise ValueError("unexpected persisted fields")


def _string(data: dict[str, Any], field: str, *, nonempty: bool = False) -> str:
    value = data[field]
    if not isinstance(value, str) or (nonempty and not value.strip()):
        raise ValueError(f"invalid {field}")
    return value


def _number(data: dict[str, Any], field: str, *, optional: bool = False) -> float | None:
    value = data[field]
    if optional and value is None:
        return None
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"invalid {field}")
    return float(value)


def prompt_to_dict(prompt: AgentPrompt) -> dict[str, Any]:
    """Encode the complete server-side prompt, validating the persisted contract."""
    data = {
        "id": prompt.id,
        "kind": prompt.kind.value,
        "workspace_id": prompt.workspace_id,
        "user_id": prompt.user_id,
        "session_id": prompt.session_id,
        "run_id": prompt.run_id,
        "tool_call_id": prompt.tool_call_id,
        "request": prompt.request,
        "status": prompt.status.value,
        "snapshot": prompt.snapshot,
        "response": prompt.response,
        "created_at": prompt.created_at,
        "expires_at": prompt.expires_at,
        "resolved_at": prompt.resolved_at,
        "resolved_by": prompt.resolved_by,
        "close_reason": prompt.close_reason,
    }
    prompt_from_dict(data)
    return _copy(data)


def prompt_from_dict(data: dict[str, Any]) -> AgentPrompt:
    """Decode a validated independent prompt, including authenticated ownership."""
    _fields(
        data,
        {
            "id",
            "kind",
            "workspace_id",
            "user_id",
            "session_id",
            "run_id",
            "tool_call_id",
            "request",
            "status",
            "snapshot",
            "response",
            "created_at",
            "expires_at",
            "resolved_at",
            "resolved_by",
            "close_reason",
        },
    )
    if not isinstance(data["request"], dict):
        raise ValueError("prompt request must be an object")
    if data["response"] is not None and not isinstance(data["response"], dict):
        raise ValueError("prompt response must be an object or null")
    kind = PromptKind(data["kind"])
    if data["response"] is not None:
        try:
            _VALIDATORS[kind](data["response"])
        except InvalidPromptResponse as exc:
            raise ValueError("invalid persisted prompt response") from exc
    created_at = _number(data, "created_at")
    assert created_at is not None
    return AgentPrompt(
        id=_string(data, "id", nonempty=True),
        kind=kind,
        workspace_id=_string(data, "workspace_id", nonempty=True),
        user_id=_string(data, "user_id", nonempty=True),
        session_id=_string(data, "session_id"),
        run_id=_string(data, "run_id"),
        tool_call_id=_string(data, "tool_call_id"),
        request=_copy(data["request"]),
        status=PromptStatus(data["status"]),
        snapshot=_string(data, "snapshot"),
        response=None if data["response"] is None else _copy(data["response"]),
        created_at=created_at,
        expires_at=_number(data, "expires_at", optional=True),
        resolved_at=_number(data, "resolved_at", optional=True),
        resolved_by=_string(data, "resolved_by"),
        close_reason=_string(data, "close_reason"),
    )


def _validate_prompt_update(before: AgentPrompt, after: AgentPrompt) -> None:
    for field in (
        "id",
        "kind",
        "workspace_id",
        "user_id",
        "session_id",
        "run_id",
        "tool_call_id",
        "request",
        "snapshot",
        "created_at",
        "expires_at",
    ):
        if getattr(before, field) != getattr(after, field):
            raise ValueError(f"prompt {field} is immutable")
    if before.response is not None:
        for field in ("response", "resolved_at", "resolved_by"):
            if getattr(before, field) != getattr(after, field):
                raise ValueError(f"answered prompt {field} is immutable")


class JsonlPromptStore:
    """Durable PromptStore; every read and compare-and-set uses a file lock."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def _read(self, journal: FileJournal, prompt_id: str | None = None) -> AgentPrompt | None:
        current: AgentPrompt | None = None
        try:
            for record in journal.records:
                if record["kind"] not in ("prompt_created", "prompt_updated"):
                    raise ValueError("unknown prompt record")
                if (current is None) != (record["kind"] == "prompt_created"):
                    raise ValueError("invalid prompt record order")
                row = prompt_from_dict(record["data"])
                if current is not None:
                    _validate_prompt_update(current, row)
                if prompt_id is not None and row.id != prompt_id:
                    raise ValueError("prompt id does not match journal")
                if hashlib.sha256(row.id.encode("utf-8")).hexdigest() != journal.path.stem:
                    raise ValueError("prompt id does not match filename")
                current = row
        except (KeyError, TypeError, ValueError, UnicodeError, OverflowError) as exc:
            raise CorruptJournalError("invalid persisted prompt") from exc
        return current

    def create(self, prompt: AgentPrompt) -> None:
        try:
            data = prompt_to_dict(prompt)
            prompt_from_dict(data)
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError) as exc:
            raise PersistenceError("invalid prompt") from exc
        with FileJournal(self.root, "prompts", prompt.id) as journal:
            if self._read(journal, prompt.id) is not None:
                raise ValueError(f"prompt {prompt.id} already exists")
            journal.append("prompt_created", data)

    def get(self, prompt_id: str) -> AgentPrompt | None:
        with FileJournal(self.root, "prompts", prompt_id) as journal:
            return self._read(journal, prompt_id)

    def transition(
        self, prompt_id: str, from_statuses: Iterable[PromptStatus], **changes: Any
    ) -> AgentPrompt | None:
        allowed = set(from_statuses)
        with FileJournal(self.root, "prompts", prompt_id) as journal:
            current = self._read(journal, prompt_id)
            if current is None or current.status not in allowed:
                return None
            try:
                updated = replace(current, **changes)
                _validate_prompt_update(current, updated)
                data = prompt_to_dict(updated)
                updated = prompt_from_dict(data)
            except (AttributeError, KeyError, TypeError, ValueError, OverflowError) as exc:
                raise PersistenceError("invalid prompt transition") from exc
            journal.append("prompt_updated", data)
            return updated

    def list_by_status(self, status: PromptStatus) -> list[AgentPrompt]:
        rows: list[AgentPrompt] = []
        for path in sorted((self.root / "prompts").glob("*.jsonl")):
            with FileJournal._from_token(self.root, "prompts", path.stem) as journal:
                row = self._read(journal)
                if row is not None and row.status is status:
                    rows.append(row)
        return rows


def _memory_to_dict(item: MemoryItem) -> dict[str, Any]:
    return {
        "id": item.id,
        "kind": item.kind.value,
        "content": item.content,
        "topic": item.topic,
        "normalized_key": item.normalized_key,
        "importance": item.importance,
        "status": item.status.value,
        "created_at": item.created_at,
        "expires_at": item.expires_at,
        "superseded_by": item.superseded_by,
        "source_session_id": item.source_session_id,
        "use_count": item.use_count,
        "last_used_at": item.last_used_at,
    }


def _memory_from_dict(data: dict[str, Any]) -> MemoryItem:
    _fields(
        data,
        {
            "id",
            "kind",
            "content",
            "topic",
            "normalized_key",
            "importance",
            "status",
            "created_at",
            "expires_at",
            "superseded_by",
            "source_session_id",
            "use_count",
            "last_used_at",
        },
    )
    if type(data["importance"]) is not int or not 1 <= data["importance"] <= 5:
        raise ValueError("invalid memory importance")
    if type(data["use_count"]) is not int or data["use_count"] < 0:
        raise ValueError("invalid memory use_count")
    created_at = _number(data, "created_at")
    assert created_at is not None
    return MemoryItem(
        id=_string(data, "id", nonempty=True),
        kind=MemoryKind(data["kind"]),
        content=_string(data, "content", nonempty=True),
        topic=_string(data, "topic"),
        normalized_key=_string(data, "normalized_key"),
        importance=data["importance"],
        status=MemoryStatus(data["status"]),
        created_at=created_at,
        expires_at=_number(data, "expires_at", optional=True),
        superseded_by=_string(data, "superseded_by"),
        source_session_id=_string(data, "source_session_id"),
        use_count=data["use_count"],
        last_used_at=_number(data, "last_used_at", optional=True),
    )


class JsonlMemoryStore:
    """Scope-separated journal with durable item and forgotten-content tombstones."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def _key(self, scope: MemoryScope) -> str:
        if (
            not isinstance(scope.workspace_id, str)
            or not scope.workspace_id.strip()
            or not isinstance(scope.subject_id, str)
            or not scope.subject_id.strip()
        ):
            raise PersistenceError("memory scope requires workspace and subject")
        return _canonical([scope.workspace_id, scope.subject_id]).decode("utf-8")

    def _read(
        self, journal: FileJournal, scope: MemoryScope
    ) -> tuple[dict[str, MemoryItem], set[str]]:
        items: dict[str, MemoryItem] = {}
        tombstones: set[str] = set()
        try:
            for record in journal.records:
                kind = record["kind"]
                if kind not in {"memory_put", "memory_batch", "memory_forget"}:
                    raise ValueError("unknown memory record")
                data = record["data"]
                payload = "items" if kind == "memory_batch" else "item"
                fields = {"workspace_id", "subject_id", payload}
                if kind == "memory_forget":
                    fields.add("fingerprint")
                _fields(data, fields)
                if (
                    data["workspace_id"] != scope.workspace_id
                    or data["subject_id"] != scope.subject_id
                ):
                    raise ValueError("memory scope does not match journal")
                values = [data["item"]] if payload == "item" else data["items"]
                if not isinstance(values, list) or not values:
                    raise ValueError("invalid memory batch")
                seen: set[str] = set()
                for value in values:
                    if not isinstance(value, dict):
                        raise ValueError("invalid memory item")
                    item = _memory_from_dict(value)
                    if item.id in seen:
                        raise ValueError("duplicate memory id in batch")
                    seen.add(item.id)
                    previous = items.get(item.id)
                    if previous is not None and previous.status is not MemoryStatus.ACTIVE:
                        if item.status is MemoryStatus.ACTIVE:
                            raise ValueError("persisted memory resurrects a tombstone")
                    if kind == "memory_forget":
                        fingerprint = _string(data, "fingerprint")
                        if (
                            not _TOKEN.fullmatch(fingerprint)
                            or item.status is not MemoryStatus.ARCHIVED
                            or fingerprint != memory_fingerprint(item.kind, item.content)
                            or previous is None
                            or item != replace(previous, status=MemoryStatus.ARCHIVED)
                        ):
                            raise ValueError("invalid forgotten memory tombstone")
                    if item.status is MemoryStatus.ARCHIVED:
                        # Old journals have no explicit fingerprint. Retain each
                        # archived version's fingerprint throughout replay.
                        tombstones.add(memory_fingerprint(item.kind, item.content))
                    items[item.id] = item
        except (KeyError, TypeError, ValueError, UnicodeError, OverflowError) as exc:
            raise CorruptJournalError("invalid persisted memory") from exc
        # Legacy writers could add forgotten content under a fresh ID. Upgrading
        # must not make that content injectable merely because its ID differs.
        for item_id, item in items.items():
            if (
                item.status is MemoryStatus.ACTIVE
                and memory_fingerprint(item.kind, item.content) in tombstones
            ):
                items[item_id] = replace(item, status=MemoryStatus.ARCHIVED)
        return items, tombstones

    def _put(self, scope: MemoryScope, item: MemoryItem, *, update: bool) -> None:
        try:
            data = _memory_to_dict(item)
            _memory_from_dict(data)
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError) as exc:
            raise PersistenceError("invalid memory") from exc
        with FileJournal(self.root, "memories", self._key(scope)) as journal:
            items, tombstones = self._read(journal, scope)
            current = items.get(item.id)
            if update and current is None:
                return
            # A delayed replacement/usage update cannot alter a terminal row.
            if current is not None and current.status is not MemoryStatus.ACTIVE:
                return
            if (
                item.status is MemoryStatus.ACTIVE
                and memory_fingerprint(item.kind, item.content) in tombstones
            ):
                raise MemoryForgottenError("memory content was forgotten")
            journal.append(
                "memory_put",
                {"workspace_id": scope.workspace_id, "subject_id": scope.subject_id, "item": data},
            )

    def add(self, scope: MemoryScope, item: MemoryItem) -> None:
        self._put(scope, item, update=False)

    def add_with_supersession(self, scope: MemoryScope, item: MemoryItem) -> None:
        try:
            data = _memory_to_dict(item)
            _memory_from_dict(data)
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError) as exc:
            raise PersistenceError("invalid memory") from exc
        with FileJournal(self.root, "memories", self._key(scope)) as journal:
            items, tombstones = self._read(journal, scope)
            current = items.get(item.id)
            if current is not None and current.status is not MemoryStatus.ACTIVE:
                return
            if (
                item.status is MemoryStatus.ACTIVE
                and memory_fingerprint(item.kind, item.content) in tombstones
            ):
                raise MemoryForgottenError("memory content was forgotten")
            updates = []
            if item.status is MemoryStatus.ACTIVE and item.normalized_key:
                for old in items.values():
                    if (
                        old.id != item.id
                        and old.status is MemoryStatus.ACTIVE
                        and old.normalized_key == item.normalized_key
                    ):
                        updates.append(
                            _memory_to_dict(
                                replace(old, status=MemoryStatus.SUPERSEDED, superseded_by=item.id)
                            )
                        )
            updates.append(data)
            journal.append(
                "memory_batch",
                {
                    "workspace_id": scope.workspace_id,
                    "subject_id": scope.subject_id,
                    "items": updates,
                },
            )

    def update(self, scope: MemoryScope, item: MemoryItem) -> None:
        self._put(scope, item, update=True)

    def get_items(self, scope: MemoryScope, item_ids: list[str]) -> list[MemoryItem]:
        """Read current rows in scope, including their terminal/expired state."""
        with FileJournal(self.root, "memories", self._key(scope)) as journal:
            items, _ = self._read(journal, scope)
            return [items[item_id] for item_id in dict.fromkeys(item_ids) if item_id in items]

    def touch(self, scope: MemoryScope, item_ids: list[str], now: float) -> None:
        """Increment usage from current rows while holding the process lock."""
        if type(now) not in (int, float) or not math.isfinite(now):
            raise PersistenceError("invalid memory usage time")
        with FileJournal(self.root, "memories", self._key(scope)) as journal:
            items, _ = self._read(journal, scope)
            updates = []
            for item_id in dict.fromkeys(item_ids):
                item = items.get(item_id)
                if (
                    item is None
                    or item.status is not MemoryStatus.ACTIVE
                    or (item.expires_at is not None and item.expires_at <= now)
                ):
                    continue
                updates.append(
                    _memory_to_dict(
                        replace(
                            item,
                            use_count=item.use_count + 1,
                            last_used_at=(
                                now if item.last_used_at is None else max(item.last_used_at, now)
                            ),
                        )
                    )
                )
            if updates:
                journal.append(
                    "memory_batch",
                    {
                        "workspace_id": scope.workspace_id,
                        "subject_id": scope.subject_id,
                        "items": updates,
                    },
                )

    def forget(self, scope: MemoryScope, item_id: str) -> bool:
        """Atomically archive the current row and permanently revoke its content."""
        with FileJournal(self.root, "memories", self._key(scope)) as journal:
            items, _ = self._read(journal, scope)
            item = items.get(item_id)
            if item is None or item.status is MemoryStatus.ARCHIVED:
                return False
            archived = replace(item, status=MemoryStatus.ARCHIVED)
            journal.append(
                "memory_forget",
                {
                    "workspace_id": scope.workspace_id,
                    "subject_id": scope.subject_id,
                    "item": _memory_to_dict(archived),
                    "fingerprint": memory_fingerprint(item.kind, item.content),
                },
            )
            return True

    def list_items(
        self, scope: MemoryScope, status: MemoryStatus = MemoryStatus.ACTIVE
    ) -> list[MemoryItem]:
        with FileJournal(self.root, "memories", self._key(scope)) as journal:
            items, _ = self._read(journal, scope)
            return [item for item in items.values() if item.status is status]
