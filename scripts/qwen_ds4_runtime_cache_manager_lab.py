#!/usr/bin/env python3
"""
DS4-style runtime cache manager prototype for llama.cpp slot save/restore.

This is the first "B path" step without modifying llama.cpp C++ yet. It turns
the earlier one-shot lab scripts into a small runtime policy manager:

- prompt prefix metadata/hash
- self-describing .qkv header
- llama.cpp /slots save/restore payloads
- cold / continued / evict save reasons
- disk budget eviction

It still uses llama.cpp whole-slot save bytes. That means it saves recompute and
session switching time, but it does not reduce the active KV buffer allocated by
llama.cpp for the currently live slot.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Any

from qwen_cache_key_lab import CacheMetadata, cache_key, canonical_json, sha256_bytes, sha256_text
from qwen_kv_artifact_lab import artifact_filename, read_artifact, verify_artifact, write_artifact
from qwen_workflow_policy_bench import WORKFLOWS, Workflow, build_prompt, make_prefix_metadata, make_workflow_prefix


def post_json(url: str, payload: dict[str, Any] | None, timeout: float) -> tuple[dict[str, Any], float]:
    body = b"" if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    return json.loads(data.decode("utf-8")), time.perf_counter() - start


def slot_file_path(slot_save_path: str, filename: str) -> Path:
    return Path(f"{slot_save_path}{filename}").expanduser()


def write_slot_payload(slot_save_path: str, filename: str, payload: bytes) -> Path:
    path = slot_file_path(slot_save_path, filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def timings_prompt_n(response: dict[str, Any]) -> int | None:
    timings = response.get("timings")
    if isinstance(timings, dict) and isinstance(timings.get("prompt_n"), int):
        return timings["prompt_n"]
    return None


def synthetic_payload(seed: str, size: int) -> bytes:
    payload = bytearray()
    counter = 0
    while len(payload) < size:
        payload.extend(sha256_bytes(f"{seed}\0{counter}".encode("utf-8")).encode("ascii"))
        counter += 1
    return bytes(payload[:size])


class SlotRuntime:
    def erase(self) -> tuple[dict[str, Any], float]:
        raise NotImplementedError

    def restore(self, filename: str) -> tuple[dict[str, Any], float]:
        raise NotImplementedError

    def save(self, filename: str, seed: str) -> tuple[dict[str, Any], float]:
        raise NotImplementedError

    def completion(self, prompt: str, n_predict: int) -> tuple[dict[str, Any], float]:
        raise NotImplementedError


class LlamaSlotRuntime(SlotRuntime):
    def __init__(self, base_url: str, slot_id: int, slot_save_path: str, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.slot_id = slot_id
        self.slot_save_path = slot_save_path
        self.timeout = timeout

    def erase(self) -> tuple[dict[str, Any], float]:
        return post_json(f"{self.base_url}/slots/{self.slot_id}?action=erase", None, self.timeout)

    def restore(self, filename: str) -> tuple[dict[str, Any], float]:
        return post_json(
            f"{self.base_url}/slots/{self.slot_id}?action=restore",
            {"filename": filename},
            self.timeout,
        )

    def save(self, filename: str, seed: str) -> tuple[dict[str, Any], float]:
        del seed
        return post_json(
            f"{self.base_url}/slots/{self.slot_id}?action=save",
            {"filename": filename},
            self.timeout,
        )

    def completion(self, prompt: str, n_predict: int) -> tuple[dict[str, Any], float]:
        return post_json(
            f"{self.base_url}/completion",
            {
                "prompt": prompt,
                "id_slot": self.slot_id,
                "cache_prompt": True,
                "n_predict": n_predict,
                "temperature": 0,
            },
            self.timeout,
        )


class DryRunSlotRuntime(SlotRuntime):
    def __init__(self, slot_save_path: str, payload_bytes: int) -> None:
        self.slot_save_path = slot_save_path
        self.payload_bytes = payload_bytes
        self.restored_tokens = 0

    def erase(self) -> tuple[dict[str, Any], float]:
        self.restored_tokens = 0
        return {"dry_run": True, "erased": True}, 0.0

    def restore(self, filename: str) -> tuple[dict[str, Any], float]:
        size = slot_file_path(self.slot_save_path, filename).stat().st_size
        self.restored_tokens = max(1, size // 49152)
        return {"dry_run": True, "n_restored": self.restored_tokens, "n_read": size}, 0.0

    def save(self, filename: str, seed: str) -> tuple[dict[str, Any], float]:
        payload = synthetic_payload(seed, self.payload_bytes)
        write_slot_payload(self.slot_save_path, filename, payload)
        return {"dry_run": True, "n_saved": max(1, self.payload_bytes // 49152), "n_written": self.payload_bytes}, 0.0

    def completion(self, prompt: str, n_predict: int) -> tuple[dict[str, Any], float]:
        prompt_n = max(1, len(prompt.encode("utf-8")) // 6)
        if self.restored_tokens:
            prompt_n = min(prompt_n, 512)
        return {
            "dry_run": True,
            "content": "{}",
            "timings": {
                "prompt_n": prompt_n,
                "predicted_n": n_predict,
            },
        }, 0.0


class RuntimeCacheManager:
    def __init__(self, args: argparse.Namespace, runtime: SlotRuntime) -> None:
        self.args = args
        self.runtime = runtime
        self.artifact_dir = Path(args.artifact_dir)
        self.slot_save_path = args.slot_save_path
        self.index_path = Path(args.index_path) if args.index_path else self.artifact_dir / "runtime-cache-index.json"
        self.index = self.load_index()
        self.current_key: str | None = None
        self.current_metadata: CacheMetadata | None = None
        self.current_workflow_id: str | None = None
        self.current_dirty = False
        self.events: list[dict[str, Any]] = []

    def load_index(self) -> dict[str, Any]:
        if self.index_path.exists():
            return json.loads(self.index_path.read_text(encoding="utf-8"))
        return {"schema": "qwen3.5-ds4-runtime-cache-index-v1", "entries": {}}

    def save_index(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(
            json.dumps(self.index, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def artifact_path(self, metadata: CacheMetadata) -> Path:
        return self.artifact_dir / artifact_filename(metadata)

    def key_for(self, metadata: CacheMetadata) -> str:
        return cache_key(metadata)

    def record_event(self, event: str, **fields: Any) -> None:
        row = {"event": event, "unix": int(time.time()), **fields}
        self.events.append(row)

    def restore_artifact(self, metadata: CacheMetadata) -> dict[str, Any]:
        key = self.key_for(metadata)
        path = self.artifact_path(metadata)
        verify = verify_artifact(path, metadata)
        _header, payload = read_artifact(path)
        filename = f"runtime-restore-{key}.bin"
        write_slot_payload(self.slot_save_path, filename, payload)
        response, latency_s = self.runtime.restore(filename)
        restored = bool(response.get("n_restored", 0) > 0 or response.get("dry_run"))

        entry = self.index["entries"].setdefault(key, {})
        entry["hit_count"] = int(entry.get("hit_count", 0)) + 1
        entry["last_used_unix"] = int(time.time())
        entry["last_restore_latency_s"] = latency_s
        self.save_index()

        self.record_event(
            "disk_hit_restore",
            cache_key=key,
            artifact_path=str(path),
            restored=restored,
            restore_latency_s=latency_s,
            payload_size=verify["payload_size"],
        )
        return {"cache_hit": restored, "restore_latency_s": latency_s, "restore_response": response}

    def save_current(self, reason: str) -> dict[str, Any] | None:
        if self.current_metadata is None or self.current_key is None:
            return None
        metadata = self.current_metadata
        key = self.current_key
        filename = f"runtime-save-{reason}-{key}.bin"
        response, latency_s = self.runtime.save(filename, f"{key}:{reason}:{len(self.events)}")
        raw_path = slot_file_path(self.slot_save_path, filename)
        payload = raw_path.read_bytes()
        extra_header = {
            "manager_schema": "qwen3.5-ds4-runtime-cache-manager-v1",
            "save_reason": reason,
            "workflow_id": self.current_workflow_id,
            "hit_count": int(self.index["entries"].get(key, {}).get("hit_count", 0)),
        }
        updates_lookup = reason == "cold" or self.args.session_checkpoints_update_lookup
        artifact_dir = self.artifact_dir if updates_lookup else self.artifact_dir / "session-checkpoints" / reason
        artifact_path = write_artifact(artifact_dir, metadata, payload, extra_header=extra_header)
        verify = verify_artifact(artifact_path, metadata)

        entry = self.index["entries"].setdefault(key, {})
        common_entry = {
            "cache_key": key,
            "workflow_id": self.current_workflow_id,
            "prompt_prefix_sha256": metadata.prompt_prefix_sha256,
            "policy_sha256": metadata.policy_sha256,
            "last_save_reason": reason,
            "last_saved_unix": int(time.time()),
            "last_save_latency_s": latency_s,
            "save_count": int(entry.get("save_count", 0)) + 1,
        }
        if updates_lookup:
            entry.update(
                {
                    **common_entry,
                    "artifact_path": str(artifact_path),
                    "payload_size": len(payload),
                    "payload_sha256": sha256_bytes(payload),
                }
            )
        else:
            session_checkpoints = entry.setdefault("session_checkpoints", [])
            if isinstance(session_checkpoints, list):
                session_checkpoints.append(
                    {
                        **common_entry,
                        "artifact_path": str(artifact_path),
                        "payload_size": len(payload),
                        "payload_sha256": sha256_bytes(payload),
                    }
                )
        entry.setdefault("created_unix", int(time.time()))
        entry.setdefault("hit_count", 0)
        self.current_dirty = False
        self.save_index()

        self.record_event(
            f"{reason}_save",
            cache_key=key,
            workflow_id=self.current_workflow_id,
            artifact_path=str(artifact_path),
            save_latency_s=latency_s,
            payload_size=len(payload),
            updates_lookup=updates_lookup,
            slot_response=response,
        )
        self.enforce_disk_budget(skip_key=key)
        return {
            "artifact_path": str(artifact_path),
            "verify": verify,
            "save_latency_s": latency_s,
            "updates_lookup": updates_lookup,
            "response": response,
        }

    def enforce_disk_budget(self, skip_key: str | None) -> None:
        budget_bytes = int(self.args.disk_budget_mib * 1024 * 1024)
        if budget_bytes <= 0:
            return

        entries = self.index.get("entries", {})

        def existing_items() -> list[tuple[str, dict[str, Any], Path]]:
            rows = []
            for key, entry in entries.items():
                path_text = entry.get("artifact_path")
                if not isinstance(path_text, str):
                    continue
                path = Path(path_text)
                if path.exists():
                    rows.append((key, entry, path))
            return rows

        def total_size(rows: list[tuple[str, dict[str, Any], Path]]) -> int:
            return sum(path.stat().st_size for _key, _entry, path in rows)

        rows = existing_items()
        evicted = []
        while total_size(rows) > budget_bytes:
            candidates = [(key, entry, path) for key, entry, path in rows if key != skip_key]
            if not candidates:
                break

            def score(item: tuple[str, dict[str, Any], Path]) -> tuple[float, int]:
                _key, entry, path = item
                payload_size = max(1, int(entry.get("payload_size", path.stat().st_size)))
                hits = int(entry.get("hit_count", 0))
                saved = int(entry.get("last_saved_unix", entry.get("created_unix", 0)))
                value = (hits + 1) / payload_size
                return (value, saved)

            victim_key, victim_entry, victim_path = min(candidates, key=score)
            victim_size = victim_path.stat().st_size
            victim_path.unlink()
            victim_entry["evicted_unix"] = int(time.time())
            victim_entry["evicted_reason"] = "disk_budget"
            evicted.append({"cache_key": victim_key, "path": str(victim_path), "size": victim_size})
            rows = existing_items()

        if evicted:
            self.save_index()
            for item in evicted:
                self.record_event("disk_budget_evict", **item)

    def switch_to(self, workflow_id: str, metadata: CacheMetadata) -> dict[str, Any]:
        key = self.key_for(metadata)
        if self.current_key == key:
            self.record_event("live_session_reuse", cache_key=key, workflow_id=workflow_id)
            return {"source": "live", "cache_hit": True, "restore_latency_s": 0.0}

        if self.current_dirty:
            self.save_current("evict")

        self.runtime.erase()
        self.current_key = key
        self.current_metadata = metadata
        self.current_workflow_id = workflow_id
        self.current_dirty = False

        if self.artifact_path(metadata).exists():
            restore = self.restore_artifact(metadata)
            return {"source": "disk", **restore}

        self.record_event("cold_miss", cache_key=key, workflow_id=workflow_id, artifact_path=str(self.artifact_path(metadata)))
        return {"source": "cold", "cache_hit": False, "restore_latency_s": 0.0}

    def run_request(self, workflow: Workflow, metadata: CacheMetadata, prompt: str, request_index: int) -> dict[str, Any]:
        switch = self.switch_to(workflow.workflow_id, metadata)
        response, latency_s = self.runtime.completion(prompt, self.args.n_predict)
        prompt_n = timings_prompt_n(response)
        self.current_dirty = True
        key = self.key_for(metadata)

        save = None
        if switch["source"] == "cold" and len(prompt) >= self.args.min_cache_chars:
            save = self.save_current("cold")
        elif switch["source"] in {"disk", "live"} and self.args.continued_interval_requests > 0:
            entry = self.index["entries"].get(key, {})
            hits = int(entry.get("hit_count", 0))
            if hits > 0 and hits % self.args.continued_interval_requests == 0:
                save = self.save_current("continued")

        self.record_event(
            "completion",
            cache_key=key,
            workflow_id=workflow.workflow_id,
            request_index=request_index,
            source=switch["source"],
            completion_latency_s=latency_s,
            prompt_n=prompt_n,
        )
        return {
            "workflow_id": workflow.workflow_id,
            "cache_key": key,
            "source": switch["source"],
            "cache_hit": bool(switch["cache_hit"]),
            "restore_latency_s": switch["restore_latency_s"],
            "completion_latency_s": latency_s,
            "prompt_n": prompt_n,
            "save": save,
            "completion_preview": str(response.get("content", ""))[:200],
        }


def workflow_prompt_for(args: argparse.Namespace, workflow: Workflow, occurrence: int) -> tuple[str, CacheMetadata, str]:
    prefix = make_workflow_prefix(workflow, args.prefix_chars, f"{args.run_tag}-runtime-{workflow.workflow_id}")
    metadata = make_prefix_metadata(args, workflow, prefix)
    task = workflow.seed_task if occurrence == 0 else workflow.followup_task
    return prefix, metadata, build_prompt(workflow, prefix, task)


def run_lab(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    if args.clean:
        for path in [Path(args.artifact_dir), Path(args.slot_save_path)]:
            if path.exists():
                shutil.rmtree(path)
        index_path = Path(args.index_path) if args.index_path else Path(args.artifact_dir) / "runtime-cache-index.json"
        if index_path.exists():
            index_path.unlink()

    Path(args.artifact_dir).mkdir(parents=True, exist_ok=True)
    Path(args.slot_save_path).mkdir(parents=True, exist_ok=True)

    runtime: SlotRuntime
    if args.dry_run:
        runtime = DryRunSlotRuntime(args.slot_save_path, args.dry_run_payload_bytes)
    else:
        runtime = LlamaSlotRuntime(args.base_url, args.slot_id, args.slot_save_path, args.timeout)
    manager = RuntimeCacheManager(args, runtime)

    sequence = [item.strip() for item in args.sequence.split(",") if item.strip()]
    unknown = [item for item in sequence if item not in WORKFLOWS]
    if unknown:
        raise SystemExit(f"unknown workflows: {', '.join(unknown)}")

    seen_counts: dict[str, int] = {}
    requests = []
    for index, workflow_id in enumerate(sequence):
        workflow = WORKFLOWS[workflow_id]
        occurrence = seen_counts.get(workflow_id, 0)
        seen_counts[workflow_id] = occurrence + 1
        _prefix, metadata, prompt = workflow_prompt_for(args, workflow, occurrence)
        requests.append(manager.run_request(workflow, metadata, prompt, index))

    if manager.current_dirty and args.save_on_shutdown:
        manager.save_current("shutdown")

    manager.save_index()
    entries = manager.index.get("entries", {})
    artifact_paths = [Path(entry["artifact_path"]) for entry in entries.values() if isinstance(entry.get("artifact_path"), str)]
    existing_artifacts = [path for path in artifact_paths if path.exists()]
    total_artifact_bytes = sum(path.stat().st_size for path in existing_artifacts)
    hits = sum(1 for item in requests if item["cache_hit"])
    misses = sum(1 for item in requests if not item["cache_hit"])
    saves_by_reason: dict[str, int] = {}
    for event in manager.events:
        name = event["event"]
        if name.endswith("_save"):
            saves_by_reason[name.removesuffix("_save")] = saves_by_reason.get(name.removesuffix("_save"), 0) + 1

    report = {
        "schema": "qwen3.5-ds4-runtime-cache-manager-lab-v1",
        "dry_run": args.dry_run,
        "base_url": args.base_url,
        "slot_id": args.slot_id,
        "policy": {
            "sequence": sequence,
            "prefix_chars": args.prefix_chars,
            "min_cache_chars": args.min_cache_chars,
            "continued_interval_requests": args.continued_interval_requests,
            "disk_budget_mib": args.disk_budget_mib,
            "slot_save_path": args.slot_save_path,
            "artifact_dir": args.artifact_dir,
            "index_path": str(manager.index_path),
            "manager_policy_sha256": sha256_text(
                canonical_json(
                    {
                        "sequence": sequence,
                        "min_cache_chars": args.min_cache_chars,
                        "continued_interval_requests": args.continued_interval_requests,
                        "disk_budget_mib": args.disk_budget_mib,
                    }
                )
            ),
        },
        "requests": requests,
        "events": manager.events,
        "summary": {
            "ok": bool(misses >= 1 and hits >= 1 and saves_by_reason.get("cold", 0) >= 1),
            "requests": len(requests),
            "cache_hits": hits,
            "cache_misses": misses,
            "saves_by_reason": saves_by_reason,
            "disk_budget_evictions": sum(1 for event in manager.events if event["event"] == "disk_budget_evict"),
            "artifact_count": len(existing_artifacts),
            "artifact_bytes": total_artifact_bytes,
            "artifact_mib": total_artifact_bytes / 1024 / 1024,
        },
    }
    return report, 0 if report["summary"]["ok"] else 1


def print_report(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print("# Qwen DS4 Runtime Cache Manager Lab")
    print()
    print(f"- dry_run: {report['dry_run']}")
    print(f"- requests: {summary['requests']}")
    print(f"- cache_hits: {summary['cache_hits']}")
    print(f"- cache_misses: {summary['cache_misses']}")
    print(f"- saves_by_reason: {summary['saves_by_reason']}")
    print(f"- disk_budget_evictions: {summary['disk_budget_evictions']}")
    print(f"- artifact_count: {summary['artifact_count']}")
    print(f"- artifact_mib: {summary['artifact_mib']:.2f}")
    print()
    for item in report["requests"]:
        print(
            f"- {item['workflow_id']}: source={item['source']} hit={item['cache_hit']} "
            f"prompt_n={item['prompt_n']} restore={item['restore_latency_s']:.3f}s "
            f"completion={item['completion_latency_s']:.3f}s"
        )
    print()
    print(f"summary_ok: {summary['ok']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prototype a DS4-style runtime cache manager over llama.cpp slots.")
    parser.add_argument("--base-url", default="http://127.0.0.1:18180")
    parser.add_argument("--slot-id", type=int, default=0)
    parser.add_argument("--slot-save-path", default="artifacts/runtime-cache-slots/")
    parser.add_argument("--artifact-dir", default="artifacts/runtime-cache-artifacts")
    parser.add_argument("--index-path", default="")
    parser.add_argument("--trace-json", default="traces/runtime-cache-manager-lab-2026-05-15.json")
    parser.add_argument("--sequence", default="fb,translation,fb,rooming,translation,fb")
    parser.add_argument("--prefix-chars", type=int, default=6000)
    parser.add_argument("--min-cache-chars", type=int, default=1024)
    parser.add_argument("--continued-interval-requests", type=int, default=2)
    parser.add_argument("--disk-budget-mib", type=float, default=512)
    parser.add_argument("--n-predict", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--run-tag", default=str(int(time.time())))
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--save-on-shutdown", action="store_true")
    parser.add_argument(
        "--session-checkpoints-update-lookup",
        action="store_true",
        help=(
            "Allow evict/continued/shutdown saves to overwrite the lookup artifact. "
            "Keep disabled for llama.cpp whole-slot prefix cache, because dirty session checkpoints can poison prefix hits."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-payload-bytes", type=int, default=2 * 1024 * 1024)
    parser.add_argument("--model-id", default="qwen3.5:4b")
    parser.add_argument("--model-sha256", default="local-model-sha256-not-provided")
    parser.add_argument("--tokenizer-sha256", default="qwen35-tokenizer-sha256-not-provided")
    parser.add_argument("--quant-type", default="Q4_K_M")
    parser.add_argument("--runtime-version", default="llama.cpp-local-build")
    parser.add_argument("--chat-template", default="chatml")
    parser.add_argument("--context-size", type=int, default=32768)
    parser.add_argument("--lora-sha256", default="none")
    parser.add_argument("--steering-state", default="none")
    parser.add_argument("--steering-vector-sha256", default="none")
    parser.add_argument("--steering-strength", default="0")
    args = parser.parse_args()

    if not args.slot_save_path.endswith("/"):
        raise SystemExit("--slot-save-path must end with / because llama.cpp concatenates path + filename")

    report, exit_code = run_lab(args)
    print_report(report)
    if args.trace_json:
        trace_path = Path(args.trace_json)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\ntrace_json: {trace_path}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
