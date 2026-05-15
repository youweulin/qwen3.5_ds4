#!/usr/bin/env python3
"""
Prototype an external cache manager before moving anything into llama.cpp.

This is the "A path" product-validation test:

Client intent -> Python cache manager -> llama.cpp /completion + /slots

The manager computes a prefix cache key, restores an existing .qkv artifact on
hit, or runs the request and saves a new artifact on miss. It is deliberately a
lab script instead of a long-running HTTP proxy so we can measure the policy
without adding more moving parts.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
import urllib.request
from pathlib import Path
from typing import Any

from qwen_cache_key_lab import cache_key, make_metadata, sha256_bytes
from qwen_kv_artifact_lab import artifact_filename, read_artifact, verify_artifact, write_artifact
from qwen_prefix_cache_lab import build_cases


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


def slot_action(
    base_url: str,
    slot_id: int,
    action: str,
    timeout: float,
    payload: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], float]:
    return post_json(f"{base_url.rstrip('/')}/slots/{slot_id}?action={action}", payload, timeout)


def completion(base_url: str, slot_id: int, prompt: str, n_predict: int, timeout: float) -> tuple[dict[str, Any], float]:
    return post_json(
        f"{base_url.rstrip('/')}/completion",
        {
            "prompt": prompt,
            "id_slot": slot_id,
            "cache_prompt": True,
            "n_predict": n_predict,
            "temperature": 0,
        },
        timeout,
    )


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


def make_prompt(prefix_chars: int, run_tag: str, case_id: str) -> tuple[str, str]:
    cases = {case.case_id: case for case in build_cases(prefix_chars)}
    case = cases[case_id]
    prefix = (
        f"CACHE_MANAGER_PROXY_LAB_RUN={run_tag}\n"
        "這個 run tag 用來避免前一次測試的 RAM prompt cache 污染本次量測。\n\n"
        f"{case.system_prefix}"
    )
    prompt = (
        f"{prefix}\n\n"
        "回覆限制：只輸出一個 JSON object，不要 markdown，不要解釋。\n\n"
        f"USER:\n{case.user_tail}\nASSISTANT:\n"
    )
    return prefix, prompt


def make_prefix_metadata(args: argparse.Namespace, prefix_text: str):
    rope_settings = {
        "rope_type": "qwen35-mrope",
        "freq_base": 10000000.0,
        "mrope_sections": [11, 11, 10, 0],
    }
    return make_metadata(
        prefix_text=prefix_text,
        full_prompt_text=prefix_text,
        cache_scope="prefix",
        steering_state=args.steering_state,
        model_id=args.model_id,
        model_sha256=args.model_sha256,
        tokenizer_sha256=args.tokenizer_sha256,
        quant_type=args.quant_type,
        runtime_name="llama.cpp",
        runtime_version=args.runtime_version,
        chat_template=args.chat_template,
        rope_settings=rope_settings,
        context_size=args.context_size,
        kv_cache_format_version="llama.cpp-slot-save-v1",
        lora_sha256=args.lora_sha256,
        policy_sha256=args.policy_sha256,
        steering_vector_sha256=args.steering_vector_sha256,
        steering_strength=args.steering_strength,
    )


def managed_completion(
    args: argparse.Namespace,
    prompt: str,
    metadata,
    *,
    allow_save_on_miss: bool,
) -> dict[str, Any]:
    key = cache_key(metadata)
    artifact_path = Path(args.artifact_dir) / artifact_filename(metadata)
    restore_latency_s = 0.0
    restored = False
    if artifact_path.exists():
        _header, payload = read_artifact(artifact_path)
        verify_artifact(artifact_path, metadata)
        restore_filename = f"proxy-restore-{key}.bin"
        write_slot_payload(args.slot_save_path, restore_filename, payload)
        restore_response, restore_latency_s = slot_action(
            args.base_url,
            args.slot_id,
            "restore",
            args.timeout,
            {"filename": restore_filename},
        )
        restored = bool(restore_response.get("n_restored", 0) > 0)
    else:
        restore_response = None

    completion_response, completion_latency_s = completion(
        args.base_url,
        args.slot_id,
        prompt,
        args.n_predict,
        args.timeout,
    )

    saved = False
    save_latency_s = 0.0
    save_response = None
    slot_payload_size = None
    slot_payload_sha256 = None
    if not artifact_path.exists() and allow_save_on_miss:
        save_filename = f"proxy-save-{key}.bin"
        save_response, save_latency_s = slot_action(
            args.base_url,
            args.slot_id,
            "save",
            args.timeout,
            {"filename": save_filename},
        )
        raw_slot_path = slot_file_path(args.slot_save_path, save_filename)
        raw_payload = raw_slot_path.read_bytes()
        write_artifact(Path(args.artifact_dir), metadata, raw_payload)
        verify_artifact(artifact_path, metadata)
        saved = True
        slot_payload_size = len(raw_payload)
        slot_payload_sha256 = sha256_bytes(raw_payload)

    return {
        "cache_key": key,
        "artifact_path": str(artifact_path),
        "cache_hit": restored,
        "cache_saved": saved,
        "restore_latency_s": restore_latency_s,
        "completion_latency_s": completion_latency_s,
        "save_latency_s": save_latency_s,
        "total_latency_s": restore_latency_s + completion_latency_s + save_latency_s,
        "prompt_n": timings_prompt_n(completion_response),
        "completion_preview": str(completion_response.get("content", ""))[:240],
        "restore_response": restore_response,
        "save_response": save_response,
        "slot_payload_size": slot_payload_size,
        "slot_payload_sha256": slot_payload_sha256,
    }


def run_lab(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    if args.clean:
        for path in [Path(args.artifact_dir), Path(args.slot_save_path)]:
            if path.exists():
                shutil.rmtree(path)
    Path(args.artifact_dir).mkdir(parents=True, exist_ok=True)
    Path(args.slot_save_path).mkdir(parents=True, exist_ok=True)

    # Baseline uses a different run tag so previous cache-manager requests cannot
    # make the cold no-manager comparison accidentally warm.
    baseline_prefix, baseline_followup_prompt = make_prompt(
        args.prefix_chars,
        f"{args.run_tag}-baseline",
        "same_prefix_new_tail",
    )
    baseline_metadata = make_prefix_metadata(args, baseline_prefix)
    slot_action(args.base_url, args.slot_id, "erase", args.timeout)
    baseline_response, baseline_latency_s = completion(
        args.base_url,
        args.slot_id,
        baseline_followup_prompt,
        args.n_predict,
        args.timeout,
    )
    baseline = {
        "prefix_cache_key": cache_key(baseline_metadata),
        "latency_s": baseline_latency_s,
        "prompt_n": timings_prompt_n(baseline_response),
        "completion_preview": str(baseline_response.get("content", ""))[:240],
    }

    managed_prefix, managed_source_prompt = make_prompt(
        args.prefix_chars,
        f"{args.run_tag}-managed",
        "cold_fixed_a",
    )
    _managed_prefix_2, managed_followup_prompt = make_prompt(
        args.prefix_chars,
        f"{args.run_tag}-managed",
        "same_prefix_new_tail",
    )
    managed_metadata = make_prefix_metadata(args, managed_prefix)

    slot_action(args.base_url, args.slot_id, "erase", args.timeout)
    manager_miss = managed_completion(args, managed_source_prompt, managed_metadata, allow_save_on_miss=True)
    slot_action(args.base_url, args.slot_id, "erase", args.timeout)
    manager_hit = managed_completion(args, managed_followup_prompt, managed_metadata, allow_save_on_miss=False)

    baseline_latency = baseline["latency_s"]
    hit_latency = manager_hit["total_latency_s"]
    speedup = baseline_latency / hit_latency if hit_latency else None
    prompt_n_saved = None
    if isinstance(baseline["prompt_n"], int) and isinstance(manager_hit["prompt_n"], int):
        prompt_n_saved = baseline["prompt_n"] - manager_hit["prompt_n"]

    report = {
        "schema": "qwen3.5-ds4-cache-manager-proxy-lab-v1",
        "base_url": args.base_url,
        "slot_id": args.slot_id,
        "policy": {
            "prefix_chars": args.prefix_chars,
            "n_predict": args.n_predict,
            "cache_scope": "prefix",
            "artifact_dir": args.artifact_dir,
            "slot_save_path": args.slot_save_path,
        },
        "baseline_no_manager_cold_followup": baseline,
        "manager_miss_then_save": manager_miss,
        "manager_hit_restore_then_followup": manager_hit,
        "summary": {
            "ok": bool(manager_miss["cache_saved"] and manager_hit["cache_hit"]),
            "baseline_latency_s": baseline_latency,
            "manager_hit_total_latency_s": hit_latency,
            "manager_hit_restore_latency_s": manager_hit["restore_latency_s"],
            "manager_hit_completion_latency_s": manager_hit["completion_latency_s"],
            "latency_speedup_vs_cold": speedup,
            "baseline_prompt_n": baseline["prompt_n"],
            "manager_hit_prompt_n": manager_hit["prompt_n"],
            "prompt_n_saved": prompt_n_saved,
        },
    }
    return report, 0 if report["summary"]["ok"] else 1


def print_report(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print("# Qwen Cache Manager Proxy Lab")
    print()
    print(f"- baseline_latency_s: {summary['baseline_latency_s']:.3f}")
    print(f"- manager_hit_total_latency_s: {summary['manager_hit_total_latency_s']:.3f}")
    print(f"- manager_hit_restore_latency_s: {summary['manager_hit_restore_latency_s']:.3f}")
    print(f"- manager_hit_completion_latency_s: {summary['manager_hit_completion_latency_s']:.3f}")
    if summary["latency_speedup_vs_cold"] is not None:
        print(f"- latency_speedup_vs_cold: {summary['latency_speedup_vs_cold']:.2f}x")
    print(f"- baseline_prompt_n: {summary['baseline_prompt_n']}")
    print(f"- manager_hit_prompt_n: {summary['manager_hit_prompt_n']}")
    print(f"- prompt_n_saved: {summary['prompt_n_saved']}")
    print()
    print("## Manager")
    print(f"- miss_saved: {report['manager_miss_then_save']['cache_saved']}")
    print(f"- hit_restored: {report['manager_hit_restore_then_followup']['cache_hit']}")
    print(f"- artifact_path: {report['manager_hit_restore_then_followup']['artifact_path']}")
    print()
    print(f"summary_ok: {summary['ok']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure an external cache manager policy before C++ runtime work.")
    parser.add_argument("--base-url", default="http://127.0.0.1:18180")
    parser.add_argument("--slot-id", type=int, default=0)
    parser.add_argument("--slot-save-path", default="artifacts/proxy-cache-slots/")
    parser.add_argument("--artifact-dir", default="artifacts/proxy-cache-artifacts")
    parser.add_argument("--trace-json", default="traces/cache-manager-proxy-lab-2026-05-15.json")
    parser.add_argument("--prefix-chars", type=int, default=6000)
    parser.add_argument("--n-predict", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--run-tag", default=str(int(time.time())))
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--model-id", default="qwen3.5:4b")
    parser.add_argument("--model-sha256", default="local-model-sha256-not-provided")
    parser.add_argument("--tokenizer-sha256", default="qwen35-tokenizer-sha256-not-provided")
    parser.add_argument("--quant-type", default="Q4_K_M")
    parser.add_argument("--runtime-version", default="llama.cpp-local-build")
    parser.add_argument("--chat-template", default="chatml")
    parser.add_argument("--context-size", type=int, default=32768)
    parser.add_argument("--lora-sha256", default="none")
    parser.add_argument("--policy-sha256", default="none")
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
