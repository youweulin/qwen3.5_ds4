#!/usr/bin/env python3
"""
Wrap real llama.cpp slot-save bytes in the qwen3.5 DS4-style KV artifact.

This replaces the earlier synthetic payload with bytes produced by a running
llama.cpp server:

1. send a long fixed-prefix prompt to a slot
2. POST /slots/:id?action=save
3. read the raw llama.cpp slot/state file
4. wrap those exact bytes in the .qkv artifact envelope
5. verify the artifact header/hash
6. extract the payload back to a raw slot file and restore it

The current llama.cpp slot endpoint saves a whole slot state, not a clean
prefix-only KV tensor slice. That is still the first practical bridge from the
metadata artifact design to actual runtime bytes.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path
from typing import Any

from qwen_cache_key_lab import cache_key, make_metadata, sha256_bytes
from qwen_kv_artifact_lab import read_artifact, verify_artifact, write_artifact
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


def slot_file_candidates(slot_save_path: str, filename: str) -> list[Path]:
    raw_path = Path(slot_save_path).expanduser()
    return [
        raw_path / filename,
        Path(f"{slot_save_path}{filename}").expanduser(),
        raw_path.parent / f"{raw_path.name}{filename}",
    ]


def find_slot_file(slot_save_path: str, filename: str) -> Path:
    candidates = slot_file_candidates(slot_save_path, filename)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    rendered = "\n".join(f"- {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"slot file not found. checked:\n{rendered}")


def write_raw_slot_file(slot_save_path: str, filename: str, payload: bytes) -> Path:
    path = slot_file_candidates(slot_save_path, filename)[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def make_prompt(prefix_chars: int, case_id: str) -> tuple[str, str, str]:
    cases = {case.case_id: case for case in build_cases(prefix_chars)}
    case = cases[case_id]
    instruction = (
        "\n\n回覆限制：只輸出一個 JSON object，不要 markdown，不要解釋。"
        "\n請把以下固定 SOP 視為本地 worker 的長期上下文。"
    )
    prompt = f"{case.system_prefix}{instruction}\n\nUSER:\n{case.user_tail}\nASSISTANT:\n"
    return case.system_prefix, case.user_tail, prompt


def make_slot_metadata(args: argparse.Namespace, prefix_text: str, full_prompt_text: str):
    rope_settings = {
        "rope_type": "qwen35-mrope",
        "freq_base": 10000000.0,
        "mrope_sections": [11, 11, 10, 0],
    }
    return make_metadata(
        prefix_text=prefix_text,
        full_prompt_text=full_prompt_text,
        cache_scope="slot",
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


def slot_action(
    base_url: str,
    slot_id: int,
    action: str,
    timeout: float,
    payload: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], float]:
    return post_json(f"{base_url.rstrip('/')}/slots/{slot_id}?action={action}", payload, timeout)


def timings_prompt_n(response: dict[str, Any]) -> int | None:
    timings = response.get("timings")
    if isinstance(timings, dict) and isinstance(timings.get("prompt_n"), int):
        return timings["prompt_n"]
    return None


def run_lab(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    source_prefix, _source_tail, source_prompt = make_prompt(args.prefix_chars, "cold_fixed_a")
    _followup_prefix, _followup_tail, followup_prompt = make_prompt(args.prefix_chars, "same_prefix_new_tail")
    metadata = make_slot_metadata(args, source_prefix, source_prompt)

    initial_response, initial_latency_s = completion(
        args.base_url,
        args.slot_id,
        source_prompt,
        args.n_predict,
        args.timeout,
    )
    save_response, save_latency_s = slot_action(
        args.base_url,
        args.slot_id,
        "save",
        args.timeout,
        {"filename": args.slot_filename},
    )

    raw_slot_path = find_slot_file(args.slot_save_path, args.slot_filename)
    raw_payload = raw_slot_path.read_bytes()
    artifact_path = write_artifact(Path(args.artifact_dir), metadata, raw_payload)
    artifact_verify = verify_artifact(artifact_path, metadata)

    artifact_header, artifact_payload = read_artifact(artifact_path)
    extracted_slot_path = write_raw_slot_file(args.slot_save_path, args.extracted_slot_filename, artifact_payload)

    erase_response, erase_latency_s = slot_action(args.base_url, args.slot_id, "erase", args.timeout)
    restore_response, restore_latency_s = slot_action(
        args.base_url,
        args.slot_id,
        "restore",
        args.timeout,
        {"filename": args.extracted_slot_filename},
    )
    followup_response, followup_latency_s = completion(
        args.base_url,
        args.slot_id,
        followup_prompt,
        args.n_predict,
        args.timeout,
    )

    initial_prompt_n = timings_prompt_n(initial_response)
    followup_prompt_n = timings_prompt_n(followup_response)
    restored_cache_effective = (
        isinstance(initial_prompt_n, int)
        and isinstance(followup_prompt_n, int)
        and followup_prompt_n < initial_prompt_n
    )
    report = {
        "schema": "qwen3.5-ds4-llamacpp-slot-artifact-lab-v1",
        "base_url": args.base_url,
        "slot_id": args.slot_id,
        "slot_save_path": args.slot_save_path,
        "raw_slot_path": str(raw_slot_path),
        "raw_slot_size": len(raw_payload),
        "raw_slot_sha256": sha256_bytes(raw_payload),
        "artifact_path": str(artifact_path),
        "artifact_cache_key": cache_key(metadata),
        "artifact_payload_size": artifact_verify["payload_size"],
        "artifact_payload_sha256": artifact_verify["payload_sha256"],
        "artifact_header_schema": artifact_header.get("artifact_schema"),
        "extracted_slot_path": str(extracted_slot_path),
        "extracted_slot_size": len(artifact_payload),
        "extracted_slot_sha256": sha256_bytes(artifact_payload),
        "initial_completion": {
            "latency_s": initial_latency_s,
            "prompt_n": initial_prompt_n,
            "content_preview": str(initial_response.get("content", ""))[:240],
        },
        "slot_save": {
            "latency_s": save_latency_s,
            "response": save_response,
        },
        "artifact_verify": artifact_verify,
        "slot_erase": {
            "latency_s": erase_latency_s,
            "response": erase_response,
        },
        "slot_restore_from_extracted_artifact_payload": {
            "latency_s": restore_latency_s,
            "response": restore_response,
        },
        "followup_after_restore": {
            "latency_s": followup_latency_s,
            "prompt_n": followup_prompt_n,
            "content_preview": str(followup_response.get("content", ""))[:240],
        },
        "checks": {
            "raw_payload_matches_artifact_payload": raw_payload == artifact_payload,
            "artifact_payload_matches_extracted_slot": extracted_slot_path.read_bytes() == raw_payload,
            "artifact_verify_ok": artifact_verify["ok"],
            "restore_reported_tokens": restore_response.get("n_restored", 0) > 0,
            "restored_cache_effective": restored_cache_effective,
        },
    }
    ok = all(report["checks"].values())
    report["summary"] = {
        "ok": ok,
        "initial_prompt_n": initial_prompt_n,
        "followup_prompt_n_after_restore": followup_prompt_n,
        "raw_slot_mib": len(raw_payload) / 1024 / 1024,
    }
    return report, 0 if ok else 1


def print_report(report: dict[str, Any]) -> None:
    print("# Qwen llama.cpp Slot Artifact Lab")
    print()
    print(f"- raw_slot_size: {report['raw_slot_size']} bytes")
    print(f"- raw_slot_sha256: {report['raw_slot_sha256']}")
    print(f"- artifact_path: {report['artifact_path']}")
    print(f"- artifact_payload_size: {report['artifact_payload_size']} bytes")
    print(f"- extracted_slot_path: {report['extracted_slot_path']}")
    print(f"- initial_prompt_n: {report['summary']['initial_prompt_n']}")
    print(f"- followup_prompt_n_after_restore: {report['summary']['followup_prompt_n_after_restore']}")
    print()
    print("## Checks")
    for name, value in report["checks"].items():
        print(f"- {name}: {'PASS' if value else 'FAIL'}")
    print()
    print(f"summary_ok: {report['summary']['ok']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Wrap real llama.cpp slot-save bytes in a .qkv artifact.")
    parser.add_argument("--base-url", default="http://127.0.0.1:18180")
    parser.add_argument("--slot-id", type=int, default=0)
    parser.add_argument("--slot-save-path", default="artifacts/llamacpp-slots/")
    parser.add_argument("--slot-filename", default="qwen35-slot-source.bin")
    parser.add_argument("--extracted-slot-filename", default="qwen35-slot-from-qkv.bin")
    parser.add_argument("--artifact-dir", default="artifacts/llamacpp-slot-artifacts")
    parser.add_argument("--trace-json", default="traces/llamacpp-slot-artifact-lab-2026-05-15.json")
    parser.add_argument("--prefix-chars", type=int, default=6000)
    parser.add_argument("--n-predict", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=240)
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
