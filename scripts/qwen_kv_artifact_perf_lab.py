#!/usr/bin/env python3
"""
Measure the overhead of the KV artifact envelope.

This benchmarks only the artifact layer:
- synthetic KV payload generation
- header/key/digest creation
- file write
- file read + header verification + payload sha256

It does not measure real model inference or real llama.cpp KV serialization.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import time
from pathlib import Path
from typing import Any

from qwen_kv_artifact_lab import (
    make_prefix_metadata,
    synthetic_kv_payload,
    verify_artifact,
    write_artifact,
)


def parse_sizes(value: str) -> list[int]:
    sizes: list[int] = []
    for part in value.split(","):
        item = part.strip().lower()
        if not item:
            continue
        multiplier = 1
        if item.endswith("kb"):
            multiplier = 1024
            item = item[:-2]
        elif item.endswith("mb"):
            multiplier = 1024 * 1024
            item = item[:-2]
        elif item.endswith("gb"):
            multiplier = 1024 * 1024 * 1024
            item = item[:-2]
        sizes.append(int(float(item) * multiplier))
    return sizes


def mb(value: int) -> float:
    return value / (1024 * 1024)


def rate_mib_s(size: int, seconds: float) -> float | None:
    if seconds <= 0:
        return None
    return mb(size) / seconds


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def run_one(root: Path, size: int, rounds: int, prefix_chars: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metadata = make_prefix_metadata(prefix_chars)
    for idx in range(rounds):
        case_dir = root / f"{size}-bytes" / f"round-{idx}"
        case_dir.mkdir(parents=True, exist_ok=True)

        start = time.perf_counter()
        payload = synthetic_kv_payload(metadata, size)
        payload_gen_s = time.perf_counter() - start

        start = time.perf_counter()
        path = write_artifact(case_dir, metadata, payload)
        write_s = time.perf_counter() - start

        start = time.perf_counter()
        verify = verify_artifact(path, metadata)
        verify_s = time.perf_counter() - start

        rows.append(
            {
                "payload_bytes": size,
                "payload_mib": mb(size),
                "round": idx,
                "path": str(path),
                "payload_gen_s": payload_gen_s,
                "write_s": write_s,
                "verify_s": verify_s,
                "write_mib_s": rate_mib_s(size, write_s),
                "verify_mib_s": rate_mib_s(size, verify_s),
                "cache_key": verify["cache_key"],
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for size in sorted({row["payload_bytes"] for row in rows}):
        subset = [row for row in rows if row["payload_bytes"] == size]
        summary[str(size)] = {
            "payload_mib": mb(size),
            "rounds": len(subset),
            "payload_gen_avg_s": statistics.mean(row["payload_gen_s"] for row in subset),
            "write_avg_s": statistics.mean(row["write_s"] for row in subset),
            "verify_avg_s": statistics.mean(row["verify_s"] for row in subset),
            "write_mib_s_avg": statistics.mean(row["write_mib_s"] for row in subset if row["write_mib_s"] is not None),
            "verify_mib_s_avg": statistics.mean(row["verify_mib_s"] for row in subset if row["verify_mib_s"] is not None),
        }
    return summary


def print_report(summary: dict[str, Any]) -> None:
    print("# Qwen KV Artifact Perf Lab")
    print()
    for size, row in summary.items():
        print(
            f"- {size} bytes ({row['payload_mib']:.3f} MiB): "
            f"write_avg_s={fmt(row['write_avg_s'])}, verify_avg_s={fmt(row['verify_avg_s'])}, "
            f"write_mib_s={fmt(row['write_mib_s_avg'])}, verify_mib_s={fmt(row['verify_mib_s_avg'])}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark KV artifact envelope overhead.")
    parser.add_argument("--sizes", default="4kb,1mb,16mb")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--prefix-chars", type=int, default=6000)
    parser.add_argument("--artifact-dir", default="artifacts/kv-artifact-perf")
    parser.add_argument("--trace-json", default="traces/kv-artifact-perf-2026-05-15.json")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    root = Path(args.artifact_dir)
    if args.clean and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for size in parse_sizes(args.sizes):
        rows.extend(run_one(root, size, args.rounds, args.prefix_chars))
    summary = summarize(rows)
    report = {
        "schema": "qwen3.5-ds4-kv-artifact-perf-v1",
        "artifact_dir": str(root),
        "rows": rows,
        "summary": summary,
    }
    print_report(summary)
    if args.trace_json:
        trace_path = Path(args.trace_json)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\ntrace_json: {trace_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
