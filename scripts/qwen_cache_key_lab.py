#!/usr/bin/env python3
"""
Validate a DS4-style cache metadata/hash design for local Qwen workers.

This does not save or load KV tensors. It tests the safety property that must
exist before SSD KV persistence is worth implementing:

    every runtime state that can change KV semantics must change the cache key.

It also separates prefix cache keys from full request keys. A new user tail
should not invalidate a reusable fixed prefix, but it should produce a different
full request key.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from qwen_prefix_cache_lab import build_cases


KEY_DOMAIN = "qwen3.5-ds4-cache-key-v1"


@dataclass(frozen=True)
class CacheMetadata:
    schema_version: str
    cache_scope: str
    model_id: str
    model_sha256: str
    tokenizer_sha256: str
    quant_type: str
    runtime_name: str
    runtime_version: str
    chat_template_sha256: str
    rope_settings_sha256: str
    context_size: int
    kv_cache_format_version: str
    lora_sha256: str
    policy_sha256: str
    steering_state: str
    steering_vector_sha256: str
    steering_strength: str
    prompt_prefix_sha256: str
    prompt_full_sha256: str


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def canonical_json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def cache_key(metadata: CacheMetadata) -> str:
    payload = canonical_json(asdict(metadata))
    return sha256_text(f"{KEY_DOMAIN}\0{payload}")


def file_sha256(path: Path, chunk_mib: int) -> str:
    digest = hashlib.sha256()
    chunk_size = max(1, chunk_mib) * 1024 * 1024
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def make_metadata(
    *,
    prefix_text: str,
    full_prompt_text: str,
    cache_scope: str,
    steering_state: str,
    model_id: str,
    model_sha256: str,
    tokenizer_sha256: str,
    quant_type: str,
    runtime_name: str,
    runtime_version: str,
    chat_template: str,
    rope_settings: dict[str, Any],
    context_size: int,
    kv_cache_format_version: str,
    lora_sha256: str,
    policy_sha256: str,
    steering_vector_sha256: str,
    steering_strength: str,
) -> CacheMetadata:
    if cache_scope == "prefix":
        prompt_full_sha256 = "not-applicable-for-prefix-scope"
    else:
        prompt_full_sha256 = sha256_text(full_prompt_text)
    return CacheMetadata(
        schema_version="1",
        cache_scope=cache_scope,
        model_id=model_id,
        model_sha256=model_sha256,
        tokenizer_sha256=tokenizer_sha256,
        quant_type=quant_type,
        runtime_name=runtime_name,
        runtime_version=runtime_version,
        chat_template_sha256=sha256_text(chat_template),
        rope_settings_sha256=sha256_text(canonical_json(rope_settings)),
        context_size=context_size,
        kv_cache_format_version=kv_cache_format_version,
        lora_sha256=lora_sha256,
        policy_sha256=policy_sha256,
        steering_state=steering_state,
        steering_vector_sha256=steering_vector_sha256,
        steering_strength=steering_strength,
        prompt_prefix_sha256=sha256_text(prefix_text),
        prompt_full_sha256=prompt_full_sha256,
    )


def result_row(name: str, expected: bool, actual: bool, note: str) -> dict[str, Any]:
    return {
        "name": name,
        "expected_key_changed": expected,
        "actual_key_changed": actual,
        "pass": expected == actual,
        "note": note,
    }


def run_lab(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    model_sha = args.model_sha256
    if args.model_path:
        model_path = Path(args.model_path).expanduser()
        model_sha = file_sha256(model_path, args.file_chunk_mib)

    cases = {case.case_id: case for case in build_cases(args.prefix_chars)}
    base_case = cases["cold_fixed_a"]
    same_tail_case = cases["warm_fixed_a_1"]
    new_tail_case = cases["same_prefix_new_tail"]
    mutated_case = cases["mutated_middle"]

    rope_settings = {
        "rope_type": "qwen35-mrope",
        "freq_base": 10000000.0,
        "mrope_sections": [11, 11, 10, 0],
    }
    common = {
        "model_id": args.model_id,
        "model_sha256": model_sha,
        "tokenizer_sha256": args.tokenizer_sha256,
        "quant_type": args.quant_type,
        "runtime_name": args.runtime_name,
        "runtime_version": args.runtime_version,
        "chat_template": args.chat_template,
        "rope_settings": rope_settings,
        "context_size": args.context_size,
        "kv_cache_format_version": args.kv_cache_format_version,
        "lora_sha256": args.lora_sha256,
        "policy_sha256": args.policy_sha256,
        "steering_vector_sha256": args.steering_vector_sha256,
        "steering_strength": args.steering_strength,
    }

    def meta_for(case_id: str, *, cache_scope: str, steering_state: str = "none") -> CacheMetadata:
        case = cases[case_id]
        return make_metadata(
            prefix_text=case.system_prefix,
            full_prompt_text=case.system_prefix + "\n" + case.user_tail,
            cache_scope=cache_scope,
            steering_state=steering_state,
            **common,
        )

    base_prefix = meta_for("cold_fixed_a", cache_scope="prefix", steering_state="none")
    same_prefix = make_metadata(
        prefix_text=same_tail_case.system_prefix,
        full_prompt_text=same_tail_case.system_prefix + "\n" + same_tail_case.user_tail,
        cache_scope="prefix",
        steering_state="none",
        **common,
    )
    new_tail_prefix = make_metadata(
        prefix_text=new_tail_case.system_prefix,
        full_prompt_text=new_tail_case.system_prefix + "\n" + new_tail_case.user_tail,
        cache_scope="prefix",
        steering_state="none",
        **common,
    )
    base_request = meta_for("cold_fixed_a", cache_scope="request", steering_state="none")
    new_tail_request = meta_for("same_prefix_new_tail", cache_scope="request", steering_state="none")
    mutated_prefix = make_metadata(
        prefix_text=mutated_case.system_prefix,
        full_prompt_text=mutated_case.system_prefix + "\n" + mutated_case.user_tail,
        cache_scope="prefix",
        steering_state="none",
        **common,
    )

    base_key = cache_key(base_prefix)
    rows = [
        result_row(
            "identical_prefix_same_state",
            False,
            cache_key(same_prefix) != base_key,
            "Same model/runtime/prefix/steering must reuse the same prefix key.",
        ),
        result_row(
            "same_prefix_new_tail_prefix_key",
            False,
            cache_key(new_tail_prefix) != base_key,
            "A different user tail should not invalidate the reusable prefix key.",
        ),
        result_row(
            "same_prefix_new_tail_request_key",
            True,
            cache_key(new_tail_request) != cache_key(base_request),
            "A full request key must include the user tail.",
        ),
        result_row(
            "mutated_prefix_text",
            True,
            cache_key(mutated_prefix) != base_key,
            "Changing the middle of the prefix must invalidate the prefix key.",
        ),
    ]

    sensitivity_cases = [
        ("model_sha256", replace(base_prefix, model_sha256="changed-model-sha")),
        ("tokenizer_sha256", replace(base_prefix, tokenizer_sha256="changed-tokenizer-sha")),
        ("quant_type", replace(base_prefix, quant_type="UD-Q4_K_XL")),
        ("runtime_version", replace(base_prefix, runtime_version="llama.cpp-b9999")),
        ("chat_template", replace(base_prefix, chat_template_sha256=sha256_text("different-template"))),
        ("rope_settings", replace(base_prefix, rope_settings_sha256=sha256_text("different-rope"))),
        ("context_size", replace(base_prefix, context_size=args.context_size * 2)),
        ("kv_cache_format_version", replace(base_prefix, kv_cache_format_version="kv-v2")),
        ("lora_sha256", replace(base_prefix, lora_sha256="lora-sha")),
        ("policy_sha256", replace(base_prefix, policy_sha256="policy-sha")),
        ("hidden_steering_state", replace(base_prefix, steering_state="alpha")),
        ("steering_vector_sha256", replace(base_prefix, steering_vector_sha256="vector-sha")),
        ("steering_strength", replace(base_prefix, steering_strength="0.75")),
    ]
    for name, variant in sensitivity_cases:
        rows.append(
            result_row(
                name,
                True,
                cache_key(variant) != base_key,
                f"Changing {name} must produce a different cache key.",
            )
        )

    report = {
        "schema": KEY_DOMAIN,
        "base_prefix_key": base_key,
        "base_request_key": cache_key(base_request),
        "base_metadata": asdict(base_prefix),
        "results": rows,
        "summary": {
            "total": len(rows),
            "passed": sum(1 for row in rows if row["pass"]),
            "failed": sum(1 for row in rows if not row["pass"]),
        },
    }
    return report, 0 if report["summary"]["failed"] == 0 else 1


def print_report(report: dict[str, Any]) -> None:
    print("# Qwen DS4 Cache Key Lab")
    print()
    for row in report["results"]:
        status = "PASS" if row["pass"] else "FAIL"
        changed = "changed" if row["actual_key_changed"] else "same"
        expected = "changed" if row["expected_key_changed"] else "same"
        print(f"- {row['name']}: {status} actual={changed} expected={expected}")
    print()
    print("## Summary")
    print(f"- total: {report['summary']['total']}")
    print(f"- passed: {report['summary']['passed']}")
    print(f"- failed: {report['summary']['failed']}")
    print(f"- base_prefix_key: {report['base_prefix_key']}")
    print(f"- base_request_key: {report['base_request_key']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate DS4-style cache metadata/hash sensitivity.")
    parser.add_argument("--prefix-chars", type=int, default=6000)
    parser.add_argument("--model-id", default="qwen3.5:4b")
    parser.add_argument("--model-path", help="Optional local GGUF path to hash fully.")
    parser.add_argument("--model-sha256", default="local-model-sha256-not-provided")
    parser.add_argument("--file-chunk-mib", type=int, default=64)
    parser.add_argument("--tokenizer-sha256", default="qwen35-tokenizer-sha256-not-provided")
    parser.add_argument("--quant-type", default="Q4_K_M")
    parser.add_argument("--runtime-name", default="llama.cpp")
    parser.add_argument("--runtime-version", default="llama.cpp-local-build")
    parser.add_argument("--chat-template", default="chatml")
    parser.add_argument("--context-size", type=int, default=32768)
    parser.add_argument("--kv-cache-format-version", default="llama.cpp-qwen35-hybrid-kv-v1")
    parser.add_argument("--lora-sha256", default="none")
    parser.add_argument("--policy-sha256", default="none")
    parser.add_argument("--steering-vector-sha256", default="none")
    parser.add_argument("--steering-strength", default="0")
    parser.add_argument("--trace-json", default="traces/cache-key-lab.json")
    args = parser.parse_args()

    report, exit_code = run_lab(args)
    print_report(report)
    if args.trace_json:
        path = Path(args.trace_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\ntrace_json: {path}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
