#!/usr/bin/env python3
"""
Measure memory behavior for the external workflow cache policy.

This launches llama-server, records its process RSS at important lifecycle
points, and compares that with the slot artifacts written to SSD. On Apple
Silicon, GPU memory is unified memory, so the script also parses llama.cpp's
startup memory breakdown as the stable Metal/KV/context reference.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from qwen_cache_key_lab import cache_key, sha256_bytes
from qwen_kv_artifact_lab import artifact_filename, read_artifact, verify_artifact, write_artifact
from qwen_workflow_policy_bench import (
    WORKFLOWS,
    build_prompt,
    completion,
    make_prefix_metadata,
    make_workflow_prefix,
    slot_action,
    slot_file_path,
    timings_prompt_n,
    write_slot_payload,
)


DEFAULT_LLAMA_SERVER = "/Users/kevinlin911/DS4/build/llama-qwen35-metal/bin/llama-server"
DEFAULT_MODEL_PATH = (
    "/Users/kevinlin911/.ollama/models/blobs/"
    "sha256-81fb60c7daa80fc1123380b98970b320ae233409f0f71a72ed7b9b0d62f40490"
)


def get_text(url: str, timeout: float) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def wait_for_server(base_url: str, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    last_error: str | None = None
    while time.time() < deadline:
        try:
            get_text(f"{base_url.rstrip('/')}/metrics", 2)
            return
        except Exception as exc:
            last_error = str(exc)
            time.sleep(0.5)
    raise RuntimeError(f"server did not become ready: {last_error}")


def start_server(args: argparse.Namespace) -> tuple[subprocess.Popen, list[str]]:
    Path(args.slot_save_path).mkdir(parents=True, exist_ok=True)
    cmd = [
        args.llama_server,
        "--alias",
        args.model_id,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--ctx-size",
        str(args.context_size),
        "--parallel",
        "1",
        "--threads",
        "-1",
        "--threads-batch",
        "-1",
        "--batch-size",
        str(args.batch_size),
        "--ubatch-size",
        str(args.ubatch_size),
        "--temp",
        "0",
        "--model",
        args.model_path,
        "--reasoning",
        "off",
        "--reasoning-budget",
        "0",
        "--reasoning-format",
        "none",
        "--chat-template",
        args.chat_template,
        "--cache-prompt",
        "--cache-reuse",
        str(args.cache_reuse),
        "--cache-ram",
        str(args.cache_ram),
        "--cache-idle-slots",
        "--ctx-checkpoints",
        str(args.ctx_checkpoints),
        "--checkpoint-every-n-tokens",
        str(args.checkpoint_every_n_tokens),
        "--slot-save-path",
        args.slot_save_path,
        "--slots",
        "--metrics",
        "--flash-attn",
        "auto",
        "--gpu-layers",
        "auto",
    ]
    logs: list[str] = []
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def reader() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            logs.append(line.rstrip())

    threading.Thread(target=reader, daemon=True).start()
    wait_for_server(f"http://{args.host}:{args.port}", args.server_start_timeout)
    return process, logs


def stop_server(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def rss_bytes(pid: int) -> int | None:
    try:
        output = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True)
    except Exception:
        return None
    value = output.strip()
    if not value:
        return None
    return int(value) * 1024


def parse_metrics(base_url: str, timeout: float) -> dict[str, float]:
    try:
        text = get_text(f"{base_url.rstrip('/')}/metrics", timeout)
    except Exception:
        return {}
    selected: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        if "kv" not in line.lower() and "slot" not in line.lower() and "prompt" not in line.lower():
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        name = parts[0]
        try:
            selected[name] = float(parts[-1])
        except ValueError:
            continue
    return selected


def parse_static_memory(logs: list[str]) -> dict[str, float]:
    patterns = {
        "metal_model_mib": r"MTL0_Mapped model buffer size =\s+([0-9.]+) MiB",
        "cpu_model_mib": r"CPU_Mapped model buffer size =\s+([0-9.]+) MiB",
        "metal_kv_mib": r"MTL0 KV buffer size =\s+([0-9.]+) MiB",
        "metal_recurrent_mib": r"MTL0 RS buffer size =\s+([0-9.]+) MiB",
        "metal_compute_mib": r"MTL0 compute buffer size =\s+([0-9.]+) MiB",
        "cpu_compute_mib": r"CPU compute buffer size =\s+([0-9.]+) MiB",
    }
    result: dict[str, float] = {}
    text = "\n".join(logs)
    for name, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            result[name] = float(match.group(1))
    if result:
        result["metal_static_total_mib"] = sum(
            result.get(name, 0.0)
            for name in ["metal_model_mib", "metal_kv_mib", "metal_recurrent_mib", "metal_compute_mib"]
        )
    return result


def sample_memory(
    label: str,
    pid: int,
    base_url: str,
    timeout: float,
    *,
    artifact_path: Path | None = None,
) -> dict[str, Any]:
    rss = rss_bytes(pid)
    artifact_size = artifact_path.stat().st_size if artifact_path and artifact_path.exists() else None
    return {
        "label": label,
        "time_unix": time.time(),
        "rss_bytes": rss,
        "rss_mib": None if rss is None else rss / 1024 / 1024,
        "artifact_path": None if artifact_path is None else str(artifact_path),
        "artifact_bytes": artifact_size,
        "artifact_mib": None if artifact_size is None else artifact_size / 1024 / 1024,
        "metrics": parse_metrics(base_url, timeout),
    }


def save_slot_artifact(args: argparse.Namespace, metadata, filename: str) -> dict[str, Any]:
    response, latency_s = slot_action(
        args.base_url,
        args.slot_id,
        "save",
        args.timeout,
        {"filename": filename},
    )
    raw_path = slot_file_path(args.slot_save_path, filename)
    payload = raw_path.read_bytes()
    artifact_path = write_artifact(Path(args.artifact_dir), metadata, payload)
    verify_artifact(artifact_path, metadata)
    return {
        "response": response,
        "latency_s": latency_s,
        "raw_slot_path": str(raw_path),
        "raw_slot_size": len(payload),
        "raw_slot_sha256": sha256_bytes(payload),
        "artifact_path": str(artifact_path),
    }


def restore_slot_artifact(args: argparse.Namespace, metadata, filename: str) -> dict[str, Any]:
    key = cache_key(metadata)
    artifact_path = Path(args.artifact_dir) / artifact_filename(metadata)
    _header, payload = read_artifact(artifact_path)
    verify_artifact(artifact_path, metadata)
    restored_slot_path = write_slot_payload(args.slot_save_path, filename, payload)
    response, latency_s = slot_action(
        args.base_url,
        args.slot_id,
        "restore",
        args.timeout,
        {"filename": filename},
    )
    return {
        "cache_key": key,
        "artifact_path": str(artifact_path),
        "restored_slot_path": str(restored_slot_path),
        "response": response,
        "latency_s": latency_s,
    }


def run_sequence(args: argparse.Namespace, process: subprocess.Popen, logs: list[str]) -> dict[str, Any]:
    pid = process.pid
    base_url = args.base_url
    samples: list[dict[str, Any]] = []
    samples.append(sample_memory("server_ready_baseline", pid, base_url, args.metrics_timeout))

    workflow_a = WORKFLOWS[args.workflow_a]
    workflow_b = WORKFLOWS[args.workflow_b]

    prefix_a = make_workflow_prefix(workflow_a, args.prefix_chars, f"{args.run_tag}-workflow-a")
    metadata_a = make_prefix_metadata(args, workflow_a, prefix_a)
    prompt_a_seed = build_prompt(workflow_a, prefix_a, workflow_a.seed_task)
    prompt_a_followup = build_prompt(workflow_a, prefix_a, workflow_a.followup_task)
    key_a = cache_key(metadata_a)

    prefix_b = make_workflow_prefix(workflow_b, args.prefix_chars, f"{args.run_tag}-workflow-b")
    prompt_b_seed = build_prompt(workflow_b, prefix_b, workflow_b.seed_task)

    slot_action(base_url, args.slot_id, "erase", args.timeout)
    completion_a, completion_a_latency = completion(base_url, args.slot_id, prompt_a_seed, args.n_predict, args.timeout)
    samples.append(sample_memory("after_workflow_a_completion", pid, base_url, args.metrics_timeout))

    save_a = save_slot_artifact(args, metadata_a, f"memory-save-a-{key_a}.bin")
    artifact_a = Path(save_a["artifact_path"])
    samples.append(sample_memory("after_workflow_a_save_artifact", pid, base_url, args.metrics_timeout, artifact_path=artifact_a))

    erase_a, erase_a_latency = slot_action(base_url, args.slot_id, "erase", args.timeout)
    samples.append(sample_memory("after_erase_workflow_a_slot", pid, base_url, args.metrics_timeout, artifact_path=artifact_a))

    completion_b, completion_b_latency = completion(base_url, args.slot_id, prompt_b_seed, args.n_predict, args.timeout)
    samples.append(sample_memory("after_workflow_b_completion", pid, base_url, args.metrics_timeout, artifact_path=artifact_a))

    restore_a = restore_slot_artifact(args, metadata_a, f"memory-restore-a-{key_a}.bin")
    samples.append(sample_memory("after_restore_workflow_a_slot", pid, base_url, args.metrics_timeout, artifact_path=artifact_a))

    completion_a_followup, completion_a_followup_latency = completion(
        base_url,
        args.slot_id,
        prompt_a_followup,
        args.n_predict,
        args.timeout,
    )
    samples.append(sample_memory("after_workflow_a_followup_hit", pid, base_url, args.metrics_timeout, artifact_path=artifact_a))

    return {
        "static_memory_from_llama_log": parse_static_memory(logs),
        "workflow_a": args.workflow_a,
        "workflow_b": args.workflow_b,
        "workflow_a_cache_key": key_a,
        "workflow_a_completion": {
            "latency_s": completion_a_latency,
            "prompt_n": timings_prompt_n(completion_a),
        },
        "workflow_a_save": save_a,
        "workflow_a_erase": {
            "latency_s": erase_a_latency,
            "response": erase_a,
        },
        "workflow_b_completion": {
            "latency_s": completion_b_latency,
            "prompt_n": timings_prompt_n(completion_b),
        },
        "workflow_a_restore": restore_a,
        "workflow_a_followup_after_restore": {
            "latency_s": completion_a_followup_latency,
            "prompt_n": timings_prompt_n(completion_a_followup),
        },
        "memory_samples": samples,
    }


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    samples = report["memory_samples"]
    rss_values = [sample["rss_mib"] for sample in samples if sample["rss_mib"] is not None]
    artifact_mib = next((sample["artifact_mib"] for sample in samples if sample["artifact_mib"] is not None), None)
    baseline_rss = samples[0]["rss_mib"]
    after_erase = next((sample["rss_mib"] for sample in samples if sample["label"] == "after_erase_workflow_a_slot"), None)
    after_restore = next((sample["rss_mib"] for sample in samples if sample["label"] == "after_restore_workflow_a_slot"), None)
    return {
        "rss_min_mib": min(rss_values) if rss_values else None,
        "rss_max_mib": max(rss_values) if rss_values else None,
        "rss_delta_max_mib": (max(rss_values) - min(rss_values)) if rss_values else None,
        "baseline_rss_mib": baseline_rss,
        "after_erase_rss_mib": after_erase,
        "after_restore_rss_mib": after_restore,
        "workflow_a_artifact_mib": artifact_mib,
        "interpretation": (
            "llama.cpp preallocates the active ctx/KV buffers, so erase/save does not make RSS collapse. "
            "The artifact moves reusable workflow state to SSD and avoids keeping multiple workflow states active at once."
        ),
    }


def run_lab(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    Path(args.artifact_dir).mkdir(parents=True, exist_ok=True)
    Path(args.slot_save_path).mkdir(parents=True, exist_ok=True)
    if not args.slot_save_path.endswith("/"):
        raise SystemExit("--slot-save-path must end with / because llama.cpp concatenates path + filename")
    args.base_url = f"http://{args.host}:{args.port}"

    process: subprocess.Popen | None = None
    logs: list[str] = []
    try:
        process, logs = start_server(args)
        report = {
            "schema": "qwen3.5-ds4-memory-policy-bench-v1",
            "base_url": args.base_url,
            "server_pid": process.pid,
            "policy": {
                "workflow_a": args.workflow_a,
                "workflow_b": args.workflow_b,
                "prefix_chars": args.prefix_chars,
                "n_predict": args.n_predict,
                "context_size": args.context_size,
                "slot_save_path": args.slot_save_path,
                "artifact_dir": args.artifact_dir,
            },
            **run_sequence(args, process, logs),
        }
        report["summary"] = summarize(report)
        return report, 0
    finally:
        stop_server(process)


def print_report(report: dict[str, Any]) -> None:
    print("# Qwen Memory Policy Bench")
    print()
    static = report["static_memory_from_llama_log"]
    if static:
        print("## Static llama.cpp memory")
        for key in sorted(static):
            print(f"- {key}: {static[key]:.2f} MiB")
        print()
    print("## Samples")
    for sample in report["memory_samples"]:
        rss = sample["rss_mib"]
        artifact = sample["artifact_mib"]
        rss_text = "n/a" if rss is None else f"{rss:.2f} MiB"
        artifact_text = "n/a" if artifact is None else f"{artifact:.2f} MiB"
        print(f"- {sample['label']}: rss={rss_text}, artifact={artifact_text}")
    print()
    summary = report["summary"]
    print("## Summary")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"- {key}: {value:.2f}")
        else:
            print(f"- {key}: {value}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure memory behavior for workflow cache save/erase/restore.")
    parser.add_argument("--llama-server", default=DEFAULT_LLAMA_SERVER)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18182)
    parser.add_argument("--slot-id", type=int, default=0)
    parser.add_argument("--slot-save-path", default="artifacts/memory-policy-slots/")
    parser.add_argument("--artifact-dir", default="artifacts/memory-policy-artifacts")
    parser.add_argument("--trace-json", default="traces/memory-policy-bench-2026-05-15.json")
    parser.add_argument("--workflow-a", choices=sorted(WORKFLOWS), default="fb")
    parser.add_argument("--workflow-b", choices=sorted(WORKFLOWS), default="translation")
    parser.add_argument("--prefix-chars", type=int, default=6000)
    parser.add_argument("--n-predict", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--metrics-timeout", type=float, default=2)
    parser.add_argument("--server-start-timeout", type=float, default=120)
    parser.add_argument("--run-tag", default=str(int(time.time())))
    parser.add_argument("--model-id", default="qwen3.5:4b")
    parser.add_argument("--model-sha256", default="local-model-sha256-not-provided")
    parser.add_argument("--tokenizer-sha256", default="qwen35-tokenizer-sha256-not-provided")
    parser.add_argument("--quant-type", default="Q4_K_M")
    parser.add_argument("--runtime-version", default="llama.cpp-local-build")
    parser.add_argument("--chat-template", default="chatml")
    parser.add_argument("--context-size", type=int, default=32768)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--ubatch-size", type=int, default=512)
    parser.add_argument("--cache-reuse", type=int, default=256)
    parser.add_argument("--cache-ram", type=int, default=8192)
    parser.add_argument("--ctx-checkpoints", type=int, default=32)
    parser.add_argument("--checkpoint-every-n-tokens", type=int, default=2048)
    parser.add_argument("--lora-sha256", default="none")
    parser.add_argument("--steering-state", default="none")
    parser.add_argument("--steering-vector-sha256", default="none")
    parser.add_argument("--steering-strength", default="0")
    args = parser.parse_args()

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
