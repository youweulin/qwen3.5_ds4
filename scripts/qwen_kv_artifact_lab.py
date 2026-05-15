#!/usr/bin/env python3
"""
Prototype SSD KV artifact naming and header verification.

This is the next step after qwen_cache_key_lab.py:

1. derive a metadata cache key
2. use that key in the KV artifact filename
3. write a self-describing header before the KV bytes
4. verify filename, header metadata, expected metadata, and payload digest

The payload here is synthetic bytes, not real llama.cpp KV tensors. The point is
to prove the safety envelope before wiring the format to real KV save/load code.
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from qwen_cache_key_lab import (
    KEY_DOMAIN,
    CacheMetadata,
    cache_key,
    canonical_json,
    make_metadata,
    sha256_bytes,
)
from qwen_prefix_cache_lab import build_cases


MAGIC = b"QWEN35DS4KV1\0"
HEADER_LEN_STRUCT = struct.Struct(">Q")
ARTIFACT_SUFFIX = ".qkv"


class ArtifactError(RuntimeError):
    pass


def artifact_filename(metadata: CacheMetadata) -> str:
    return f"{metadata.cache_scope}-{cache_key(metadata)}{ARTIFACT_SUFFIX}"


def build_header(metadata: CacheMetadata, payload: bytes) -> dict[str, Any]:
    key = cache_key(metadata)
    return {
        "artifact_schema": "qwen3.5-ds4-kv-artifact-v1",
        "key_domain": KEY_DOMAIN,
        "cache_key": key,
        "cache_scope": metadata.cache_scope,
        "metadata": asdict(metadata),
        "payload_sha256": sha256_bytes(payload),
        "payload_size": len(payload),
        "created_unix": int(time.time()),
        "created_by": "qwen_kv_artifact_lab.py",
    }


def write_artifact(directory: Path, metadata: CacheMetadata, payload: bytes) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    header = build_header(metadata, payload)
    header_bytes = canonical_json(header).encode("utf-8")
    path = directory / artifact_filename(metadata)
    with path.open("wb") as handle:
        handle.write(MAGIC)
        handle.write(HEADER_LEN_STRUCT.pack(len(header_bytes)))
        handle.write(header_bytes)
        handle.write(payload)
    return path


def read_artifact(path: Path) -> tuple[dict[str, Any], bytes]:
    data = path.read_bytes()
    if not data.startswith(MAGIC):
        raise ArtifactError("bad_magic")
    offset = len(MAGIC)
    if len(data) < offset + HEADER_LEN_STRUCT.size:
        raise ArtifactError("missing_header_length")
    header_len = HEADER_LEN_STRUCT.unpack(data[offset : offset + HEADER_LEN_STRUCT.size])[0]
    offset += HEADER_LEN_STRUCT.size
    header_end = offset + header_len
    if len(data) < header_end:
        raise ArtifactError("truncated_header")
    try:
        header = json.loads(data[offset:header_end].decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"bad_header_json:{exc}") from exc
    return header, data[header_end:]


def metadata_from_header(header: dict[str, Any]) -> CacheMetadata:
    metadata = header.get("metadata")
    if not isinstance(metadata, dict):
        raise ArtifactError("missing_metadata")
    try:
        return CacheMetadata(**metadata)
    except TypeError as exc:
        raise ArtifactError(f"bad_metadata:{exc}") from exc


def verify_artifact(path: Path, expected_metadata: CacheMetadata | None = None) -> dict[str, Any]:
    header, payload = read_artifact(path)
    metadata = metadata_from_header(header)
    computed_key = cache_key(metadata)
    header_key = header.get("cache_key")
    filename_key = path.name.removesuffix(ARTIFACT_SUFFIX).split("-", 1)[-1]
    payload_digest = sha256_bytes(payload)

    checks = {
        "filename_suffix": path.name.endswith(ARTIFACT_SUFFIX),
        "header_key_matches_metadata": header_key == computed_key,
        "filename_key_matches_header": filename_key == header_key,
        "payload_sha256_matches": header.get("payload_sha256") == payload_digest,
        "payload_size_matches": header.get("payload_size") == len(payload),
    }
    if expected_metadata is not None:
        checks["expected_metadata_key_matches"] = cache_key(expected_metadata) == computed_key
    ok = all(checks.values())
    if not ok:
        failed = [name for name, value in checks.items() if not value]
        raise ArtifactError(",".join(failed))
    return {
        "ok": True,
        "path": str(path),
        "cache_key": computed_key,
        "cache_scope": metadata.cache_scope,
        "payload_size": len(payload),
        "payload_sha256": payload_digest,
        "checks": checks,
    }


def make_prefix_metadata(prefix_chars: int, steering_state: str = "none") -> CacheMetadata:
    cases = {case.case_id: case for case in build_cases(prefix_chars)}
    case = cases["cold_fixed_a"]
    rope_settings = {
        "rope_type": "qwen35-mrope",
        "freq_base": 10000000.0,
        "mrope_sections": [11, 11, 10, 0],
    }
    return make_metadata(
        prefix_text=case.system_prefix,
        full_prompt_text=case.system_prefix + "\n" + case.user_tail,
        cache_scope="prefix",
        steering_state=steering_state,
        model_id="qwen3.5:4b",
        model_sha256="local-model-sha256-not-provided",
        tokenizer_sha256="qwen35-tokenizer-sha256-not-provided",
        quant_type="Q4_K_M",
        runtime_name="llama.cpp",
        runtime_version="llama.cpp-local-build",
        chat_template="chatml",
        rope_settings=rope_settings,
        context_size=32768,
        kv_cache_format_version="llama.cpp-qwen35-hybrid-kv-v1",
        lora_sha256="none",
        policy_sha256="none",
        steering_vector_sha256="none",
        steering_strength="0",
    )


def synthetic_kv_payload(metadata: CacheMetadata, size: int) -> bytes:
    seed = canonical_json(asdict(metadata)).encode("utf-8")
    chunks: list[bytes] = []
    counter = 0
    while sum(len(chunk) for chunk in chunks) < size:
        chunks.append(sha256_bytes(seed + str(counter).encode("ascii")).encode("ascii"))
        counter += 1
    return b"".join(chunks)[:size]


def write_tampered_header(path: Path, target: Path) -> None:
    header, payload = read_artifact(path)
    metadata = dict(header["metadata"])
    metadata["context_size"] = metadata["context_size"] * 2
    header["metadata"] = metadata
    header_bytes = canonical_json(header).encode("utf-8")
    with target.open("wb") as handle:
        handle.write(MAGIC)
        handle.write(HEADER_LEN_STRUCT.pack(len(header_bytes)))
        handle.write(header_bytes)
        handle.write(payload)


def run_case(name: str, expected_ok: bool, fn) -> dict[str, Any]:
    try:
        result = fn()
        actual_ok = True
        error = None
    except ArtifactError as exc:
        result = None
        actual_ok = False
        error = str(exc)
    return {
        "name": name,
        "expected_ok": expected_ok,
        "actual_ok": actual_ok,
        "pass": expected_ok == actual_ok,
        "error": error,
        "result": result,
    }


def run_lab(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    artifact_dir = Path(args.artifact_dir)
    if args.clean and artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    metadata = make_prefix_metadata(args.prefix_chars)
    payload = synthetic_kv_payload(metadata, args.payload_bytes)
    valid_path = write_artifact(artifact_dir / "valid", metadata, payload)

    wrong_name_dir = artifact_dir / "filename-mismatch"
    wrong_name_dir.mkdir(parents=True, exist_ok=True)
    wrong_name_path = wrong_name_dir / f"{metadata.cache_scope}-{'0' * 64}{ARTIFACT_SUFFIX}"
    shutil.copyfile(valid_path, wrong_name_path)

    tampered_payload_dir = artifact_dir / "payload-tamper"
    tampered_payload_dir.mkdir(parents=True, exist_ok=True)
    tampered_payload_path = tampered_payload_dir / valid_path.name
    shutil.copyfile(valid_path, tampered_payload_path)
    with tampered_payload_path.open("r+b") as handle:
        handle.seek(-1, 2)
        last = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([last[0] ^ 0xFF]))

    tampered_header_dir = artifact_dir / "header-tamper"
    tampered_header_dir.mkdir(parents=True, exist_ok=True)
    tampered_header_path = tampered_header_dir / valid_path.name
    write_tampered_header(valid_path, tampered_header_path)

    wrong_expected_metadata = replace(metadata, steering_state="alpha")

    rows = [
        run_case("valid_artifact", True, lambda: verify_artifact(valid_path, metadata)),
        run_case("filename_key_mismatch", False, lambda: verify_artifact(wrong_name_path, metadata)),
        run_case("payload_tamper", False, lambda: verify_artifact(tampered_payload_path, metadata)),
        run_case("header_metadata_tamper", False, lambda: verify_artifact(tampered_header_path, metadata)),
        run_case("expected_metadata_mismatch", False, lambda: verify_artifact(valid_path, wrong_expected_metadata)),
    ]
    report = {
        "schema": "qwen3.5-ds4-kv-artifact-lab-v1",
        "artifact_dir": str(artifact_dir),
        "valid_artifact": str(valid_path),
        "base_cache_key": cache_key(metadata),
        "payload_bytes": args.payload_bytes,
        "results": rows,
        "summary": {
            "total": len(rows),
            "passed": sum(1 for row in rows if row["pass"]),
            "failed": sum(1 for row in rows if not row["pass"]),
        },
    }
    return report, 0 if report["summary"]["failed"] == 0 else 1


def print_report(report: dict[str, Any]) -> None:
    print("# Qwen KV Artifact Lab")
    print()
    for row in report["results"]:
        status = "PASS" if row["pass"] else "FAIL"
        print(
            f"- {row['name']}: {status} actual_ok={row['actual_ok']} "
            f"expected_ok={row['expected_ok']}"
        )
        if row["error"]:
            print(f"  error={row['error']}")
    print()
    print("## Summary")
    print(f"- total: {report['summary']['total']}")
    print(f"- passed: {report['summary']['passed']}")
    print(f"- failed: {report['summary']['failed']}")
    print(f"- base_cache_key: {report['base_cache_key']}")
    print(f"- valid_artifact: {report['valid_artifact']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prototype KV artifact filename/header verification.")
    parser.add_argument("--prefix-chars", type=int, default=6000)
    parser.add_argument("--payload-bytes", type=int, default=4096)
    parser.add_argument("--artifact-dir", default="artifacts/kv-cache-lab")
    parser.add_argument("--trace-json", default="traces/kv-artifact-lab-2026-05-15.json")
    parser.add_argument("--clean", action="store_true", help="Remove the artifact directory before running.")
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
