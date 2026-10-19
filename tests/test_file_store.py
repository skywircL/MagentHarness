"""Durable local journals, process-safe approval CAS and isolated memory."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from support import EDITOR, VIEWER, build, run

from magent_harness import RunRequest, RunStatus
from magent_harness.file_store import (
    CorruptJournalError,
    FileJournal,
    JsonlMemoryStore,
    JsonlPromptStore,
    PersistenceError,
    StoreBusy,
    prompt_from_dict,
    prompt_to_dict,
)
from magent_harness.hitl import AgentPrompt, PromptBroker, PromptKind, PromptStatus
from magent_harness.memory import (
    MemoryForgottenError,
    MemoryItem,
    MemoryKind,
    MemoryScope,
    MemoryService,
    MemoryStatus,
    memory_fingerprint,
)
from magent_harness.testing import reply


def prompt(prompt_id: str = "p1") -> AgentPrompt:
    return AgentPrompt(
        id=prompt_id,
        kind=PromptKind.TOOL_APPROVAL,
        workspace_id=EDITOR.workspace_id,
        user_id=EDITOR.user_id,
        session_id="s1",
        run_id="r1",
        tool_call_id="t1",
        request={"arguments": {"content": "修改内容"}},
        snapshot="server-side snapshot",
        created_at=1760000000.1234567,
    )


def test_journal_commits_hash_chain_and_reopens_independent_data(tmp_path: Path) -> None:
    data = {"nested": {"ids": ["a"]}}
    with FileJournal(tmp_path, "runs", "key") as journal:
        first = journal.append("started", data)
        data["nested"]["ids"].append("modified")
        first["data"]["nested"]["ids"].append("returned")
        second = journal.append("finished", {"answer": "完成"})
        records = journal.records
        records[0]["data"]["nested"]["ids"].append("records")
        assert journal.seq == 2
        assert second["prev_hash"] == journal.records[0]["digest"]
        assert journal.digest == second["digest"]
        assert journal.records[0]["data"] == {"nested": {"ids": ["a"]}}
        assert journal.path.stat().st_mode & 0o777 == 0o600
        assert journal.path.parent.stat().st_mode & 0o777 == 0o700
    with FileJournal(tmp_path, "runs", "key") as fresh:
        assert fresh.seq == 2
        assert fresh.records[0]["data"] == {"nested": {"ids": ["a"]}}
        assert fresh.records[1]["data"]["answer"] == "完成"


@pytest.mark.parametrize("tail", [b'{"version":', b"\xff\xe4", b'{"valid":"but not committed"}'])
def test_only_unterminated_tail_is_repaired(tmp_path: Path, tail: bytes) -> None:
    with FileJournal(tmp_path, "runs", "r") as journal:
        journal.append("started", {"value": "preserved"})
        path = journal.path
    committed = path.read_bytes()
    with path.open("ab") as file:
        file.write(tail)
    with FileJournal(tmp_path, "runs", "r") as repaired:
        assert repaired.seq == 1
        assert repaired.path.read_bytes() == committed
        repaired.append("continued", {})
    with FileJournal(tmp_path, "runs", "r") as reopened:
        assert reopened.seq == 2


def _rehash(record: dict[str, Any]) -> None:
    content = {key: value for key, value in record.items() if key != "digest"}
    encoded = json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    record["digest"] = hashlib.sha256(encoded.encode()).hexdigest()


@pytest.mark.parametrize("mutation", ["version", "sequence", "chain", "digest", "data", "kind"])
def test_committed_corruption_is_rejected_without_truncation(tmp_path: Path, mutation: str) -> None:
    with FileJournal(tmp_path, "runs", "r") as journal:
        journal.append("first", {})
        second = journal.append("second", {"value": 1})
        path = journal.path
    if mutation == "version":
        second["version"] = 99
    elif mutation == "sequence":
        second["seq"] = True
    elif mutation == "chain":
        second["prev_hash"] = "wrong"
    elif mutation == "digest":
        second["digest"] = "wrong"
    elif mutation == "data":
        second["data"] = []
    else:
        second["kind"] = ""
    if mutation != "digest":
        _rehash(second)
    first = path.read_bytes().split(b"\n")[0]
    corrupted = first + b"\n" + json.dumps(second).encode() + b"\n" + b"unfinished"
    path.write_bytes(corrupted)
    with pytest.raises(CorruptJournalError):
        with FileJournal(tmp_path, "runs", "r"):
            pass
    assert path.read_bytes() == corrupted


@pytest.mark.parametrize("raw", [b"bad\n", b"\n", b'{"version":1,"version":1}\n', b"NaN\n"])
def test_complete_invalid_json_is_rejected(tmp_path: Path, raw: bytes) -> None:
    with FileJournal(tmp_path, "runs", "r") as journal:
        path = journal.path
    path.write_bytes(raw)
    with pytest.raises(CorruptJournalError):
        with FileJournal(tmp_path, "runs", "r"):
            pass


@pytest.mark.parametrize("value", [float("nan"), float("inf"), {1: "integer key"}, object()])
def test_non_json_appends_fail_before_writing(tmp_path: Path, value: Any) -> None:
    with FileJournal(tmp_path, "runs", "r") as journal:
        with pytest.raises(PersistenceError):
            journal.append("invalid", {"value": value})
        assert journal.seq == 0
        assert journal.path.read_bytes() == b""


def test_nonblocking_lock_and_released_lock(tmp_path: Path) -> None:
    with FileJournal(tmp_path, "runs", "r"):
        with pytest.raises(StoreBusy):
            with FileJournal(tmp_path, "runs", "r", blocking=False):
                pass
    with FileJournal(tmp_path, "runs", "r", blocking=False) as journal:
        journal.append("released", {})


def test_journal_lock_survives_await(tmp_path: Path) -> None:
    async def check() -> None:
        with FileJournal(tmp_path, "runs", "r") as journal:
            await asyncio.sleep(0)
            with pytest.raises(StoreBusy):
                with FileJournal(tmp_path, "runs", "r", blocking=False):
                    pass
            journal.append("still locked", {})

    asyncio.run(check())


def test_package_import_works_without_posix_but_file_store_fails_explicitly(tmp_path: Path) -> None:
    source = """
import builtins
import sys
original_import = builtins.__import__
def portable_import(name, *args, **kwargs):
    if name == 'fcntl':
        raise ImportError('simulated non-POSIX platform')
    return original_import(name, *args, **kwargs)
builtins.__import__ = portable_import
import magent_harness
assert magent_harness.PromptBroker().store is not None
try:
    with magent_harness.FileJournal(sys.argv[1], 'runs', 'r'):
        raise AssertionError('file store must reject unsupported platform')
except magent_harness.PersistenceError as error:
    assert 'POSIX' in str(error)
"""
    root = tmp_path / "unused"
    result = subprocess.run(
        [sys.executable, "-c", source, str(root)], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stderr
    assert not root.exists()


def test_every_new_directory_is_private(tmp_path: Path) -> None:
    root = tmp_path / "new" / "nested" / "root"
    with FileJournal(root, "runs", "r") as journal:
        journal.append("created", {})
    for path in (tmp_path / "new", tmp_path / "new" / "nested", root, root / "runs"):
        assert path.stat().st_mode & 0o777 == 0o700


def test_keys_are_hashed_and_namespaces_cannot_traverse(tmp_path: Path) -> None:
    with FileJournal(tmp_path, "runs", "../../outside\0/秘密") as journal:
        journal.append("safe", {})
        assert journal.path.parent == tmp_path / "runs"
        assert len(journal.path.stem) == 64
    for namespace in ("..", "../outside", "/tmp", "runs/other", ""):
        with pytest.raises(ValueError):
            FileJournal(tmp_path, namespace, "r")


def test_namespace_and_journal_symlinks_are_rejected(tmp_path: Path) -> None:
    destination = tmp_path / "outside"
    destination.mkdir()
    (tmp_path / "runs").symlink_to(destination, target_is_directory=True)
    with pytest.raises(PersistenceError):
        with FileJournal(tmp_path, "runs", "r"):
            pass
    (tmp_path / "runs").unlink()
    with FileJournal(tmp_path, "runs", "r") as journal:
        path = journal.path
    path.unlink()
    outside_file = destination / "data"
    outside_file.write_bytes(b"preserved")
    path.symlink_to(outside_file)
    with pytest.raises(PersistenceError):
        with FileJournal(tmp_path, "runs", "r"):
            pass
    assert outside_file.read_bytes() == b"preserved"


def test_checkpoint_binding_corruption_and_missing_are_disposable(tmp_path: Path) -> None:
    with FileJournal(tmp_path, "runs", "r") as journal:
        assert journal.read_checkpoint() is None
        journal.append("started", {})
        journal.write_checkpoint({"state": ["one"]})
        checkpoint = journal.read_checkpoint()
        assert checkpoint == {"state": ["one"]}
        assert journal.checkpoint_path.stat().st_mode & 0o777 == 0o600
        checkpoint["state"].append("mutation")
        assert journal.read_checkpoint() == {"state": ["one"]}
        journal.append("new", {})
        assert journal.read_checkpoint() is None
        journal.write_checkpoint({"state": "new"})
        journal.checkpoint_path.write_bytes(b"broken")
        assert journal.read_checkpoint() is None
        journal.checkpoint_path.unlink()
        assert journal.read_checkpoint() is None
        assert journal.seq == 2


def test_failed_checkpoint_replace_preserves_previous_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with FileJournal(tmp_path, "runs", "r") as journal:
        journal.append("started", {})
        journal.write_checkpoint({"old": True})

        def fail_replace(*args: Any) -> None:
            raise OSError("simulated crash before rename")

        monkeypatch.setattr(os, "replace", fail_replace)
        with pytest.raises(PersistenceError):
            journal.write_checkpoint({"new": True})
        assert journal.read_checkpoint() == {"old": True}
        assert sorted(path.name for path in journal.path.parent.iterdir()) == sorted(
            [journal.path.name, journal.checkpoint_path.name]
        )


def test_append_io_failure_closes_stale_journal_and_requires_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with FileJournal(tmp_path, "runs", "r") as journal:
        journal.append("before", {})
        with monkeypatch.context() as patch:

            def fail_sync(descriptor: int) -> None:
                raise OSError("simulated fsync failure")

            patch.setattr(os, "fsync", fail_sync)
            with pytest.raises(PersistenceError, match="reopen"):
                journal.append("uncertain", {})
        with pytest.raises(PersistenceError, match="context manager"):
            journal.append("must not use stale sequence", {})
    with FileJournal(tmp_path, "runs", "r") as recovered:
        assert [record["kind"] for record in recovered.records] == ["before", "uncertain"]
        recovered.append("replayed first", {})
        assert recovered.seq == 3


def test_partial_write_and_repeated_close_error_preserve_committed_prefix(tmp_path: Path) -> None:
    class FailedFile:
        def __init__(self, actual: Any) -> None:
            self.actual = actual

        def seek(self, *args: Any) -> int:
            return self.actual.seek(*args)

        def write(self, data: bytes) -> int:
            self.actual.write(data[:5])
            self.actual.flush()
            raise OSError("simulated partial write")

        def close(self) -> None:
            self.actual.close()
            raise OSError("simulated repeated flush failure during close")

    with FileJournal(tmp_path, "runs", "r") as journal:
        journal.append("committed", {})
        journal._file = FailedFile(journal._file)
        with pytest.raises(PersistenceError, match="reopen"):
            journal.append("torn", {})
    with FileJournal(tmp_path, "runs", "r", blocking=False) as recovered:
        assert [record["kind"] for record in recovered.records] == ["committed"]


def test_prompt_roundtrip_status_listing_and_no_mutable_aliases(tmp_path: Path) -> None:
    original = prompt()
    store = JsonlPromptStore(tmp_path)
    store.create(original)
    original.request["arguments"]["content"] = "changed input"
    fresh = JsonlPromptStore(tmp_path)
    restored = fresh.get(original.id)
    assert restored is not None
    assert restored.request["arguments"]["content"] == "修改内容"
    restored.request["arguments"]["content"] = "changed output"
    assert fresh.get(original.id).request["arguments"]["content"] == "修改内容"
    assert fresh.get("missing") is None
    assert [row.id for row in fresh.list_by_status(PromptStatus.SUSPENDED)] == [original.id]
    with pytest.raises(ValueError, match="already exists"):
        fresh.create(prompt())
    answered = fresh.transition(
        original.id,
        [PromptStatus.SUSPENDED],
        status=PromptStatus.ANSWERED,
        response={"approved": True},
    )
    assert answered is not None and answered.approved
    assert (
        store.transition(original.id, [PromptStatus.SUSPENDED], status=PromptStatus.CANCELED)
        is None
    )
    assert fresh.list_by_status(PromptStatus.SUSPENDED) == []
    assert fresh.list_by_status(PromptStatus.ANSWERED)[0].id == original.id


def test_prompt_cas_race_between_real_processes(tmp_path: Path) -> None:
    JsonlPromptStore(tmp_path).create(prompt())
    source = """
import sys
from magent_harness.file_store import JsonlPromptStore
from magent_harness.hitl import PromptStatus
store = JsonlPromptStore(sys.argv[1])
print('ready', flush=True)
sys.stdin.readline()
result = store.transition('p1', [PromptStatus.SUSPENDED],
                         status=PromptStatus.ANSWERED, response={'approved': True})
print('won' if result is not None else 'lost', flush=True)
"""
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", source, str(tmp_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]
    try:
        for process in processes:
            assert process.stdout.readline().strip() == "ready"
        for process in processes:
            process.stdin.write("go\n")
            process.stdin.flush()
        outcomes = []
        for process in processes:
            stdout, stderr = process.communicate(timeout=10)
            assert process.returncode == 0, stderr
            outcomes.append(stdout.strip())
        assert sorted(outcomes) == ["lost", "won"]
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=10)


def test_broker_uses_fresh_disk_rows_and_claims_once(tmp_path: Path) -> None:
    JsonlPromptStore(tmp_path).create(prompt())
    broker = PromptBroker(JsonlPromptStore(tmp_path))
    row = broker.resolve(EDITOR, "p1", {"approved": True})
    claimed = PromptBroker(JsonlPromptStore(tmp_path)).claim(row)
    assert claimed.status is PromptStatus.EXECUTING
    broker.mark_executed(row.id)
    assert JsonlPromptStore(tmp_path).get(row.id).status is PromptStatus.DONE


@pytest.mark.parametrize(
    "field,value",
    [
        ("workspace_id", ""),
        ("created_at", float("nan")),
        ("status", "unknown"),
        ("response", {"approved": "yes"}),
        ("request", []),
    ],
)
def test_prompt_contract_rejects_invalid_persisted_fields(field: str, value: Any) -> None:
    data = prompt_to_dict(prompt())
    data[field] = value
    with pytest.raises((ValueError, TypeError)):
        prompt_from_dict(data)


def test_semantically_invalid_prompt_is_rejected_despite_valid_hash(tmp_path: Path) -> None:
    data = prompt_to_dict(prompt())
    data["workspace_id"] = ""
    with FileJournal(tmp_path, "prompts", "p1") as journal:
        journal.append("prompt_created", data)
    with pytest.raises(CorruptJournalError):
        JsonlPromptStore(tmp_path).get("p1")


@pytest.mark.parametrize(
    "field,value",
    [
        ("id", "different"),
        ("kind", PromptKind.ASK_USER),
        ("workspace_id", "different"),
        ("user_id", "different"),
        ("session_id", "different"),
        ("run_id", "different"),
        ("tool_call_id", "different"),
        ("request", {"changed": True}),
        ("snapshot", "different"),
        ("created_at", 42.0),
        ("expires_at", 42.0),
    ],
)
def test_prompt_cas_cannot_rebind_initial_metadata(tmp_path: Path, field: str, value: Any) -> None:
    store = JsonlPromptStore(tmp_path)
    original = prompt()
    store.create(original)
    with pytest.raises(PersistenceError):
        store.transition(original.id, [PromptStatus.SUSPENDED], **{field: value})
    assert store.get(original.id) == original


def test_replay_rejects_rebinding_prompt_metadata_even_with_valid_hash(tmp_path: Path) -> None:
    original = prompt()
    JsonlPromptStore(tmp_path).create(original)
    data = prompt_to_dict(replace(original, user_id="different"))
    with FileJournal(tmp_path, "prompts", original.id) as journal:
        journal.append("prompt_updated", data)
    with pytest.raises(CorruptJournalError):
        JsonlPromptStore(tmp_path).get(original.id)


@pytest.mark.parametrize(
    "changes",
    [
        {"response": {"approved": False}},
        {"response": None},
        {"resolved_at": 200.0},
        {"resolved_by": "different"},
    ],
)
def test_prompt_answer_is_immutable_after_resolution(
    tmp_path: Path, changes: dict[str, Any]
) -> None:
    store = JsonlPromptStore(tmp_path)
    store.create(prompt())
    broker = PromptBroker(store, clock=lambda: 100.0)
    answered = broker.resolve(EDITOR, "p1", {"approved": True})
    with pytest.raises(PersistenceError):
        store.transition("p1", [PromptStatus.ANSWERED], **changes)
    assert JsonlPromptStore(tmp_path).get("p1") == answered


def test_memory_survives_new_instances_and_retains_lifecycle(tmp_path: Path) -> None:
    service = MemoryService(JsonlMemoryStore(tmp_path), clock=lambda: 100)
    old = service.remember(EDITOR, MemoryKind.FACT, "Uses MySQL", topic="Database")
    new = service.remember(EDITOR, MemoryKind.FACT, "Uses Postgres", topic="database")
    scope = MemoryScope.for_caller(EDITOR)
    store = JsonlMemoryStore(tmp_path)
    assert [item.id for item in store.list_items(scope)] == [new.id]
    assert store.list_items(scope, MemoryStatus.SUPERSEDED)[0].superseded_by == new.id
    assert store.list_items(scope, MemoryStatus.SUPERSEDED)[0].id == old.id
    assert MemoryService(store).forget(EDITOR, new.id)
    assert JsonlMemoryStore(tmp_path).list_items(scope) == []
    assert JsonlMemoryStore(tmp_path).list_items(scope, MemoryStatus.ARCHIVED)[0].id == new.id
    # Delayed recall usage updates and duplicate stale add cannot undo tombstones.
    store.update(scope, replace(new, use_count=1, last_used_at=101))
    store.add(scope, new)
    store.update(scope, old)
    assert JsonlMemoryStore(tmp_path).list_items(scope) == []


def test_memory_batch_reads_legacy_put_and_commits_replacement_in_one_record(
    tmp_path: Path,
) -> None:
    scope = MemoryScope.for_caller(EDITOR)
    store = JsonlMemoryStore(tmp_path)
    old = MemoryItem("old", MemoryKind.FACT, "Uses MySQL", normalized_key="database")
    store.add(scope, old)
    new = MemoryService(store).remember(EDITOR, MemoryKind.FACT, "Uses Postgres", topic="Database")
    fresh = JsonlMemoryStore(tmp_path)
    assert [item.id for item in fresh.list_items(scope)] == [new.id]
    assert fresh.list_items(scope, MemoryStatus.SUPERSEDED)[0].id == old.id
    path = next((tmp_path / "memories").glob("*.jsonl"))
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["kind"] for record in records] == ["memory_put", "memory_batch"]
    assert len(records[-1]["data"]["items"]) == 2


@pytest.mark.parametrize("after_commit", [False, True])
def test_memory_replacement_failure_preserves_complete_old_or_new_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, after_commit: bool
) -> None:
    service = MemoryService(JsonlMemoryStore(tmp_path))
    old = service.remember(EDITOR, MemoryKind.FACT, "Uses MySQL", topic="database")
    append = FileJournal.append

    def interrupted_append(journal: FileJournal, kind: str, data: dict[str, Any]) -> dict[str, Any]:
        assert kind == "memory_batch"
        if after_commit:
            append(journal, kind, data)
        raise PersistenceError("simulated crash at atomic replacement commit")

    with monkeypatch.context() as patch:
        patch.setattr(FileJournal, "append", interrupted_append)
        with pytest.raises(PersistenceError):
            service.remember(EDITOR, MemoryKind.FACT, "Uses Postgres", topic="database")
    scope = MemoryScope.for_caller(EDITOR)
    fresh = JsonlMemoryStore(tmp_path)
    active = fresh.list_items(scope)
    assert len(active) == 1
    superseded = fresh.list_items(scope, MemoryStatus.SUPERSEDED)
    if after_commit:
        assert active[0].content == "Uses Postgres"
        assert [item.id for item in superseded] == [old.id]
        assert superseded[0].superseded_by == active[0].id
    else:
        assert active == [old]
        assert superseded == []


def test_uncommitted_memory_batch_cannot_partially_supersede(tmp_path: Path) -> None:
    service = MemoryService(JsonlMemoryStore(tmp_path))
    old = service.remember(EDITOR, MemoryKind.FACT, "Uses MySQL", topic="database")
    path = next((tmp_path / "memories").glob("*.jsonl"))
    with path.open("ab") as file:
        file.write(b'{"kind":"memory_batch","data":{"items":[{"status":"superseded"}')
    fresh = JsonlMemoryStore(tmp_path)
    scope = MemoryScope.for_caller(EDITOR)
    assert fresh.list_items(scope) == [old]
    assert fresh.list_items(scope, MemoryStatus.SUPERSEDED) == []


def test_memory_topic_replacement_serializes_two_processes(tmp_path: Path) -> None:
    service = MemoryService(JsonlMemoryStore(tmp_path))
    old = service.remember(EDITOR, MemoryKind.FACT, "Uses MySQL", topic="database")
    source = """
import sys
from magent_harness.file_store import JsonlMemoryStore
from magent_harness.identity import Caller, Role
from magent_harness.memory import MemoryService, MemoryKind
service = MemoryService(JsonlMemoryStore(sys.argv[1]))
print('ready', flush=True)
sys.stdin.readline()
item = service.remember(Caller('ws1', 'alice', Role.EDITOR),
                        MemoryKind.FACT, sys.argv[2], topic='database')
print(item.id, flush=True)
"""
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", source, str(tmp_path), content],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for content in ("Uses Postgres", "Uses SQLite")
    ]
    try:
        for process in processes:
            assert process.stdout.readline().strip() == "ready"
        for process in processes:
            process.stdin.write("go\n")
            process.stdin.flush()
        added_ids = []
        for process in processes:
            stdout, stderr = process.communicate(timeout=10)
            assert process.returncode == 0, stderr
            added_ids.append(stdout.strip())
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=10)
    scope = MemoryScope.for_caller(EDITOR)
    fresh = JsonlMemoryStore(tmp_path)
    active = fresh.list_items(scope)
    superseded = fresh.list_items(scope, MemoryStatus.SUPERSEDED)
    assert len(active) == 1 and active[0].id in added_ids
    assert len(superseded) == 2
    assert {item.id for item in [*active, *superseded]} == {old.id, *added_ids}


def test_memory_scope_tuple_cannot_collide_or_read_another_user(tmp_path: Path) -> None:
    store = JsonlMemoryStore(tmp_path)
    alice = MemoryScope.for_caller(EDITOR)
    item = MemoryItem("m", MemoryKind.FACT, "Private knowledge")
    store.add(alice, item)
    assert store.list_items(MemoryScope.for_caller(VIEWER)) == []
    assert store.list_items(MemoryScope("another-workspace", alice.subject_id)) == []
    left, right = MemoryScope("a/b", "c"), MemoryScope("a", "b/c")
    store.add(left, item)
    assert store.list_items(right) == []
    store.update(right, item)
    assert store.list_items(right) == []


def test_invalid_memory_is_not_committed(tmp_path: Path) -> None:
    store = JsonlMemoryStore(tmp_path)
    with pytest.raises(PersistenceError):
        store.add(
            MemoryScope.for_caller(EDITOR),
            MemoryItem("m", MemoryKind.FACT, "fact", created_at=float("nan")),
        )
    assert not (tmp_path / "memories").exists()


def test_forgotten_content_cannot_return_under_a_new_id_or_topic(tmp_path: Path) -> None:
    memory = MemoryService(JsonlMemoryStore(tmp_path))
    item = memory.remember(EDITOR, MemoryKind.PROFILE, "Works on search", source_session_id="s1")
    assert memory.forget(EDITOR, item.id)
    fresh = MemoryService(JsonlMemoryStore(tmp_path))
    with pytest.raises(MemoryForgottenError):
        fresh.remember(
            EDITOR,
            MemoryKind.PROFILE,
            "  Works\n on   search ",
            topic="new topic",
            source_session_id="s2",
        )
    assert fresh.recall(EDITOR, "search").prompt == ""
    # A forgotten personal profile does not revoke a different kind or owner.
    fresh.remember(EDITOR, MemoryKind.FACT, "Works on search")
    fresh.remember(VIEWER, MemoryKind.PROFILE, "Works on search")
    forgotten = [
        json.loads(line)
        for path in (tmp_path / "memories").glob("*.jsonl")
        for line in path.read_text().splitlines()
        if "memory_forget" in line
    ]
    assert forgotten[0]["data"]["fingerprint"] == memory_fingerprint(item.kind, item.content)


@pytest.mark.parametrize("method", ["add", "update", "add_with_supersession"])
def test_all_memory_writers_reject_forgotten_fingerprints(tmp_path: Path, method: str) -> None:
    store = JsonlMemoryStore(tmp_path)
    scope = MemoryScope.for_caller(EDITOR)
    forgotten = MemoryItem("old", MemoryKind.FACT, "Uses MySQL")
    store.add(scope, forgotten)
    assert store.forget(scope, forgotten.id)
    other = MemoryItem("new", MemoryKind.FACT, "Uses Postgres")
    store.add(scope, other)
    with pytest.raises(MemoryForgottenError):
        getattr(JsonlMemoryStore(tmp_path), method)(scope, replace(other, content="Uses MySQL"))
    assert store.list_items(scope) == [other]


def test_forget_superseded_item_revokes_content_without_changing_replacement(
    tmp_path: Path,
) -> None:
    service = MemoryService(JsonlMemoryStore(tmp_path))
    old = service.remember(EDITOR, MemoryKind.FACT, "Uses MySQL", topic="database")
    new = service.remember(EDITOR, MemoryKind.FACT, "Uses Postgres", topic="database")
    scope = MemoryScope.for_caller(EDITOR)
    assert service.store.forget(scope, old.id)
    assert not service.store.forget(scope, old.id)
    assert not service.store.forget(MemoryScope.for_caller(VIEWER), new.id)
    with pytest.raises(MemoryForgottenError):
        service.remember(EDITOR, MemoryKind.FACT, "Uses MySQL", topic="database")
    assert service.store.list_items(scope) == [new]


def test_terminal_memory_rows_cannot_be_mutated_by_stale_writers(tmp_path: Path) -> None:
    store = JsonlMemoryStore(tmp_path)
    scope = MemoryScope.for_caller(EDITOR)
    item = MemoryItem("m", MemoryKind.FACT, "Private fact")
    store.add(scope, item)
    assert store.forget(scope, item.id)
    archived = store.get_items(scope, [item.id])[0]
    store.update(scope, replace(item, content="Changed", status=MemoryStatus.SUPERSEDED))
    store.add_with_supersession(scope, replace(item, content="Changed"))
    store.touch(scope, [item.id], 100)
    assert JsonlMemoryStore(tmp_path).get_items(scope, [item.id]) == [archived]


def test_fresh_memory_reads_include_expired_and_terminal_state_and_are_scoped(
    tmp_path: Path,
) -> None:
    scope = MemoryScope.for_caller(EDITOR)
    store = JsonlMemoryStore(tmp_path)
    expired = MemoryItem("expired", MemoryKind.FACT, "Expired fact", expires_at=1)
    archived = MemoryItem("archived", MemoryKind.FACT, "Archived fact")
    store.add(scope, expired)
    store.add(scope, archived)
    assert JsonlMemoryStore(tmp_path).forget(scope, archived.id)
    current = store.get_items(scope, [archived.id, expired.id, "missing", archived.id])
    assert [item.id for item in current] == [archived.id, expired.id]
    assert current[0].status is MemoryStatus.ARCHIVED
    assert current[1].expires_at == 1
    assert store.get_items(MemoryScope.for_caller(VIEWER), [archived.id, expired.id]) == []


def test_usage_touch_updates_only_current_active_metadata(tmp_path: Path) -> None:
    scope = MemoryScope.for_caller(EDITOR)
    stale_store = JsonlMemoryStore(tmp_path)
    old = MemoryItem("m", MemoryKind.FACT, "Old text")
    stale_store.add(scope, old)
    fresh = JsonlMemoryStore(tmp_path)
    current = replace(old, content="Current text", use_count=5, last_used_at=200)
    fresh.update(scope, current)
    expired = MemoryItem("expired", MemoryKind.FACT, "Expired", expires_at=100)
    fresh.add(scope, expired)
    stale_store.touch(scope, [old.id, old.id, expired.id, "missing"], 150)
    rows = fresh.get_items(scope, [old.id, expired.id])
    assert rows == [replace(current, use_count=6), expired]
    fresh.forget(scope, old.id)
    stale_store.touch(scope, [old.id], 300)
    assert fresh.get_items(scope, [old.id])[0] == replace(rows[0], status=MemoryStatus.ARCHIVED)


def test_legacy_archive_derives_a_content_tombstone_even_after_fresh_id_revival(
    tmp_path: Path,
) -> None:
    scope = MemoryScope.for_caller(EDITOR)
    store = JsonlMemoryStore(tmp_path)
    item = MemoryItem("old", MemoryKind.PROFILE, "Legacy private profile")
    store.add(scope, item)
    path = next((tmp_path / "memories").glob("*.jsonl"))
    data = json.loads(path.read_text().splitlines()[0])["data"]
    with FileJournal._from_token(tmp_path, "memories", path.stem) as journal:
        data["item"]["status"] = "archived"
        journal.append("memory_put", data)
        data["item"]["id"] = "legacy-revival"
        data["item"]["status"] = "active"
        journal.append("memory_put", data)
    fresh = MemoryService(JsonlMemoryStore(tmp_path))
    assert fresh.recall(EDITOR, "profile").prompt == ""
    assert fresh.store.get_items(scope, ["legacy-revival"])[0].status is MemoryStatus.ARCHIVED
    with pytest.raises(MemoryForgottenError):
        fresh.remember(EDITOR, item.kind, item.content)


@pytest.mark.parametrize("after_commit", [False, True])
def test_forget_failure_preserves_complete_archive_and_tombstone_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, after_commit: bool
) -> None:
    service = MemoryService(JsonlMemoryStore(tmp_path))
    item = service.remember(EDITOR, MemoryKind.PROFILE, "Private profile")
    append = FileJournal.append

    def interrupted_append(journal: FileJournal, kind: str, data: dict[str, Any]) -> dict[str, Any]:
        assert kind == "memory_forget"
        if after_commit:
            append(journal, kind, data)
        raise PersistenceError("simulated crash at atomic forget commit")

    with monkeypatch.context() as patch:
        patch.setattr(FileJournal, "append", interrupted_append)
        with pytest.raises(PersistenceError):
            service.forget(EDITOR, item.id)
    scope = MemoryScope.for_caller(EDITOR)
    fresh = MemoryService(JsonlMemoryStore(tmp_path))
    if after_commit:
        assert fresh.store.list_items(scope) == []
        with pytest.raises(MemoryForgottenError):
            fresh.remember(EDITOR, item.kind, item.content)
    else:
        assert fresh.store.list_items(scope) == [item]
        assert [row.id for row in fresh.recall(EDITOR, "profile").items] == [item.id]


def test_uncommitted_forget_does_not_archive_or_create_a_tombstone(tmp_path: Path) -> None:
    service = MemoryService(JsonlMemoryStore(tmp_path))
    item = service.remember(EDITOR, MemoryKind.PROFILE, "Still remembered")
    path = next((tmp_path / "memories").glob("*.jsonl"))
    with path.open("ab") as file:
        file.write(b'{"kind":"memory_forget","data":{"fingerprint":"torn')
    fresh = MemoryService(JsonlMemoryStore(tmp_path))
    assert fresh.store.list_items(MemoryScope.for_caller(EDITOR)) == [item]
    fresh.remember(EDITOR, item.kind, item.content)


@pytest.mark.parametrize("damage", ["fingerprint", "scope", "status", "extra", "content", "id"])
def test_committed_forget_record_validation_fails_closed(tmp_path: Path, damage: str) -> None:
    service = MemoryService(JsonlMemoryStore(tmp_path))
    item = service.remember(EDITOR, MemoryKind.PROFILE, "Private profile")
    service.forget(EDITOR, item.id)
    path = next((tmp_path / "memories").glob("*.jsonl"))
    records = [json.loads(line) for line in path.read_text().splitlines()]
    data = records[-1]["data"]
    if damage == "fingerprint":
        data["fingerprint"] = "0" * 64
    elif damage == "scope":
        data["subject_id"] = "someone-else"
    elif damage == "status":
        data["item"]["status"] = "active"
    elif damage == "extra":
        data["unknown"] = True
    elif damage == "content":
        data["item"]["content"] = "Changed content"
        data["fingerprint"] = memory_fingerprint(item.kind, "Changed content")
    else:
        data["item"]["id"] = "unknown-item"
    _rehash(records[-1])
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    with pytest.raises(CorruptJournalError):
        JsonlMemoryStore(tmp_path).get_items(MemoryScope.for_caller(EDITOR), [item.id])


def test_run_rechecks_memory_authority_after_candidate_selection(tmp_path: Path) -> None:
    scope = MemoryScope.for_caller(EDITOR)
    seed = MemoryService(JsonlMemoryStore(tmp_path))
    item = seed.remember(EDITOR, MemoryKind.PROFILE, "Revoked private profile")

    class StaleCandidates(JsonlMemoryStore):
        def list_items(
            self, scope: MemoryScope, status: MemoryStatus = MemoryStatus.ACTIVE
        ) -> list[MemoryItem]:
            rows = super().list_items(scope, status)
            JsonlMemoryStore(self.root).forget(scope, item.id)
            return rows

    memory = MemoryService(StaleCandidates(tmp_path))
    harness = build([reply("answer")], memory=memory)
    result = run(harness.engine.run(RunRequest(EDITOR, "new-session", "hello")))
    assert result.status is RunStatus.COMPLETED
    assert "Revoked private profile" not in harness.model.calls[0].messages[0].content
    assert JsonlMemoryStore(tmp_path).get_items(scope, [item.id])[0].use_count == 0


def test_memory_usage_increments_serialize_across_processes(tmp_path: Path) -> None:
    store = JsonlMemoryStore(tmp_path)
    scope = MemoryScope.for_caller(EDITOR)
    item = MemoryItem("counter", MemoryKind.FACT, "Counter")
    store.add(scope, item)
    source = """
import sys
from magent_harness.file_store import JsonlMemoryStore
from magent_harness.memory import MemoryScope
store = JsonlMemoryStore(sys.argv[1])
scope = MemoryScope('ws1', 'user:alice')
print('ready', flush=True)
sys.stdin.readline()
for _ in range(10):
    store.touch(scope, ['counter'], 100)
"""
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", source, str(tmp_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]
    try:
        for process in processes:
            assert process.stdout.readline().strip() == "ready"
        for process in processes:
            process.stdin.write("go\n")
            process.stdin.flush()
        for process in processes:
            _, stderr = process.communicate(timeout=10)
            assert process.returncode == 0, stderr
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=10)
    assert store.get_items(scope, [item.id]) == [replace(item, use_count=20, last_used_at=100)]
