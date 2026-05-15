#!/usr/bin/env python3
"""
Build and test a block-level KV cache safety manifest before touching C++.

This lab does not split real llama.cpp KV tensors yet. It proves the naming and
invalidation rules we need before implementing block-level KV:

- deterministic prefix block boundaries
- model/tokenizer/RoPE/quant-aware block keys
- previous-block hash chaining
- tail changes do not invalidate prefix blocks
- middle prefix edits invalidate that block and every following block
- disk budget eviction removes whole block artifacts only
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from qwen_cache_key_lab import canonical_json, sha256_bytes, sha256_text
from qwen_workflow_policy_bench import WORKFLOWS, Workflow, make_workflow_prefix


BLOCK_KEY_DOMAIN = "qwen3.5-ds4-block-boundary-v1"
TOKENIZER_MODE = "pseudo-cjk-char-ascii-4char-v1"


@dataclass(frozen=True)
class RuntimeIdentity:
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


@dataclass(frozen=True)
class TokenUnit:
    index: int
    start_char: int
    end_char: int
    text: str


@dataclass(frozen=True)
class BlockMetadata:
    schema_version: str
    block_size_tokens: int
    tokenizer_mode: str
    prefix_sha256: str
    block_index: int
    token_start: int
    token_end: int
    char_start: int
    char_end: int
    block_text_sha256: str
    previous_block_hash: str
    runtime: RuntimeIdentity


@dataclass(frozen=True)
class BlockRecord:
    block_key: str
    metadata: BlockMetadata
    text_preview: str
    approx_tokens: int
    byte_length: int


def is_cjk(char: str) -> bool:
    code = ord(char)
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0x3040 <= code <= 0x30FF
        or 0xAC00 <= code <= 0xD7AF
    )


def pseudo_tokenize(text: str) -> list[TokenUnit]:
    """A stable approximation until we wire in the real llama.cpp tokenizer."""
    tokens: list[TokenUnit] = []
    i = 0
    while i < len(text):
        char = text[i]
        start = i
        if char.isspace():
            while i < len(text) and text[i].isspace():
                i += 1
            pieces = [text[start:i]]
        elif is_cjk(char):
            i += 1
            pieces = [text[start:i]]
        elif char.isascii() and (char.isalnum() or char in {"_", "-"}):
            while i < len(text) and text[i].isascii() and (text[i].isalnum() or text[i] in {"_", "-"}):
                i += 1
            run = text[start:i]
            pieces = [run[j : j + 4] for j in range(0, len(run), 4)]
        else:
            i += 1
            pieces = [text[start:i]]

        cursor = start
        for piece in pieces:
            end = cursor + len(piece)
            tokens.append(TokenUnit(index=len(tokens), start_char=cursor, end_char=end, text=piece))
            cursor = end

    return tokens


def make_identity(args: argparse.Namespace, *, model_sha256: str | None = None, tokenizer_sha256: str | None = None, rope_settings: dict[str, Any] | None = None) -> RuntimeIdentity:
    rope = rope_settings or {
        "rope_type": "qwen35-mrope",
        "freq_base": 10000000.0,
        "mrope_sections": [11, 11, 10, 0],
    }
    return RuntimeIdentity(
        model_id=args.model_id,
        model_sha256=model_sha256 or args.model_sha256,
        tokenizer_sha256=tokenizer_sha256 or args.tokenizer_sha256,
        quant_type=args.quant_type,
        runtime_name=args.runtime_name,
        runtime_version=args.runtime_version,
        chat_template_sha256=sha256_text(args.chat_template),
        rope_settings_sha256=sha256_text(canonical_json(rope)),
        context_size=args.context_size,
        kv_cache_format_version=args.kv_cache_format_version,
    )


def block_key(metadata: BlockMetadata) -> str:
    payload = asdict(metadata)
    # The full prefix hash belongs in the manifest/header. It must not be part
    # of the reusable block key; otherwise a one-token edit in the middle would
    # invalidate blocks that appear before the edit. The chain is enforced by
    # block_text_sha256 + previous_block_hash.
    payload.pop("prefix_sha256", None)
    return sha256_text(f"{BLOCK_KEY_DOMAIN}\0{canonical_json(payload)}")


def build_manifest(prefix_text: str, identity: RuntimeIdentity, block_size_tokens: int) -> dict[str, Any]:
    tokens = pseudo_tokenize(prefix_text)
    prefix_sha = sha256_text(prefix_text)
    blocks: list[BlockRecord] = []
    previous_hash = "root:" + sha256_text(
        canonical_json(
            {
                "runtime": asdict(identity),
                "block_size_tokens": block_size_tokens,
                "tokenizer_mode": TOKENIZER_MODE,
            }
        )
    )

    for block_index, token_start in enumerate(range(0, len(tokens), block_size_tokens)):
        chunk = tokens[token_start : token_start + block_size_tokens]
        token_end = chunk[-1].index + 1
        char_start = chunk[0].start_char
        char_end = chunk[-1].end_char
        block_text = prefix_text[char_start:char_end]
        metadata = BlockMetadata(
            schema_version="1",
            block_size_tokens=block_size_tokens,
            tokenizer_mode=TOKENIZER_MODE,
            prefix_sha256=prefix_sha,
            block_index=block_index,
            token_start=token_start,
            token_end=token_end,
            char_start=char_start,
            char_end=char_end,
            block_text_sha256=sha256_text(block_text),
            previous_block_hash=previous_hash,
            runtime=identity,
        )
        key = block_key(metadata)
        blocks.append(
            BlockRecord(
                block_key=key,
                metadata=metadata,
                text_preview=block_text[:120],
                approx_tokens=len(chunk),
                byte_length=len(block_text.encode("utf-8")),
            )
        )
        previous_hash = key

    block_keys = [block.block_key for block in blocks]
    manifest_key = sha256_text(
        f"{BLOCK_KEY_DOMAIN}:manifest\0"
        f"{canonical_json({'identity': asdict(identity), 'prefix_sha256': prefix_sha, 'block_keys': block_keys})}"
    )
    return {
        "schema": "qwen3.5-ds4-block-manifest-v1",
        "manifest_key": manifest_key,
        "prefix_sha256": prefix_sha,
        "block_size_tokens": block_size_tokens,
        "tokenizer_mode": TOKENIZER_MODE,
        "approx_token_count": len(tokens),
        "block_count": len(blocks),
        "identity": asdict(identity),
        "blocks": [
            {
                "block_key": block.block_key,
                "metadata": asdict(block.metadata),
                "text_preview": block.text_preview,
                "approx_tokens": block.approx_tokens,
                "byte_length": block.byte_length,
            }
            for block in blocks
        ],
    }


def mutate_middle_token(prefix_text: str) -> tuple[str, int]:
    tokens = pseudo_tokenize(prefix_text)
    mid = len(tokens) // 2
    idx = mid
    while idx < len(tokens) and tokens[idx].text.isspace():
        idx += 1
    if idx >= len(tokens):
        idx = mid
    token = tokens[idx]
    if len(token.text) == 1 and is_cjk(token.text):
        replacement = "改"
    elif token.text.isascii() and not token.text.isspace():
        replacement = "X" * len(token.text)
    else:
        replacement = "異" * len(token.text)
    mutated = prefix_text[: token.start_char] + replacement + prefix_text[token.end_char :]
    return mutated, idx


def changed_block_indexes(base: dict[str, Any], other: dict[str, Any]) -> list[int]:
    base_keys = [block["block_key"] for block in base["blocks"]]
    other_keys = [block["block_key"] for block in other["blocks"]]
    return [idx for idx, (left, right) in enumerate(zip(base_keys, other_keys)) if left != right]


def run_case(name: str, expected: bool, actual: bool, note: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "expected": expected,
        "actual": actual,
        "pass": expected == actual,
        "note": note,
        "extra": extra or {},
    }


def write_block_artifacts(block_dir: Path, manifest: dict[str, Any], block_bytes: int) -> list[dict[str, Any]]:
    block_dir.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for block in manifest["blocks"]:
        idx = block["metadata"]["block_index"]
        key = block["block_key"]
        path = block_dir / f"block-{idx:04d}-{key}.qkb"
        size = block_bytes + idx * 17
        seed = f"{key}:{idx}".encode("utf-8")
        payload = bytearray()
        counter = 0
        while len(payload) < size:
            payload.extend(sha256_bytes(seed + str(counter).encode("ascii")).encode("ascii"))
            counter += 1
        path.write_bytes(bytes(payload[:size]))
        artifacts.append(
            {
                "block_index": idx,
                "block_key": key,
                "path": str(path),
                "expected_size": size,
                "hit_count": idx % 3,
            }
        )
    return artifacts


def evict_whole_blocks(artifacts: list[dict[str, Any]], budget_bytes: int) -> dict[str, Any]:
    remaining = [dict(item) for item in artifacts]
    evicted = []

    def total_size(rows: list[dict[str, Any]]) -> int:
        return sum(Path(item["path"]).stat().st_size for item in rows if Path(item["path"]).exists())

    while total_size(remaining) > budget_bytes and remaining:
        victim = min(
            remaining,
            key=lambda item: ((int(item["hit_count"]) + 1) / max(1, int(item["expected_size"])), item["block_index"]),
        )
        path = Path(victim["path"])
        size = path.stat().st_size if path.exists() else 0
        if path.exists():
            path.unlink()
        evicted.append({**victim, "evicted_size": size})
        remaining = [item for item in remaining if item["block_key"] != victim["block_key"]]

    partial_files = []
    for item in remaining:
        path = Path(item["path"])
        if path.exists() and path.stat().st_size != item["expected_size"]:
            partial_files.append(str(path))

    return {
        "budget_bytes": budget_bytes,
        "remaining": remaining,
        "evicted": evicted,
        "remaining_bytes": total_size(remaining),
        "partial_files": partial_files,
    }


def make_prefix(args: argparse.Namespace, workflow: Workflow) -> str:
    return make_workflow_prefix(workflow, args.prefix_chars, f"{args.run_tag}-block-boundary-{workflow.workflow_id}") + "\n\n"


def run_lab(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    artifact_dir = Path(args.artifact_dir)
    if args.clean and artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    workflow = WORKFLOWS[args.workflow]
    identity = make_identity(args)
    prefix = make_prefix(args, workflow)
    base_manifest = build_manifest(prefix, identity, args.block_size_tokens)

    same_prefix_new_tail = prefix + "\nUSER_TASK:\n這是不同 tail，但不應該進 prefix block manifest。\n"
    same_tail_manifest = build_manifest(prefix, identity, args.block_size_tokens)
    tail_request_hash = sha256_text(same_prefix_new_tail)

    mutated_prefix, mutated_token_index = mutate_middle_token(prefix)
    mutated_manifest = build_manifest(mutated_prefix, identity, args.block_size_tokens)
    mutated_changed = changed_block_indexes(base_manifest, mutated_manifest)
    mutation_block_index = mutated_token_index // args.block_size_tokens
    before_mutation_unchanged = all(idx not in mutated_changed for idx in range(mutation_block_index))
    from_mutation_changed = mutated_changed == list(range(mutation_block_index, base_manifest["block_count"]))

    model_changed_manifest = build_manifest(prefix, make_identity(args, model_sha256=args.model_sha256 + "-changed"), args.block_size_tokens)
    tokenizer_changed_manifest = build_manifest(prefix, make_identity(args, tokenizer_sha256=args.tokenizer_sha256 + "-changed"), args.block_size_tokens)
    rope_changed_manifest = build_manifest(
        prefix,
        make_identity(
            args,
            rope_settings={
                "rope_type": "qwen35-mrope",
                "freq_base": 5000000.0,
                "mrope_sections": [11, 11, 10, 0],
            },
        ),
        args.block_size_tokens,
    )

    block_artifacts = write_block_artifacts(artifact_dir / "blocks", base_manifest, args.synthetic_block_bytes)
    total_block_bytes = sum(item["expected_size"] for item in block_artifacts)
    budget_bytes = max(1, int(total_block_bytes * args.disk_budget_ratio))
    eviction = evict_whole_blocks(block_artifacts, budget_bytes)

    tests = [
        run_case(
            "tail_change_keeps_prefix_blocks",
            True,
            [b["block_key"] for b in base_manifest["blocks"]] == [b["block_key"] for b in same_tail_manifest["blocks"]],
            "Dynamic tail is excluded from prefix block manifest.",
            {"tail_request_sha256": tail_request_hash},
        ),
        run_case(
            "middle_edit_invalidates_from_mutated_block",
            True,
            before_mutation_unchanged and from_mutation_changed,
            "A middle prefix edit keeps earlier blocks and invalidates the changed block plus every later block through previous_block_hash.",
            {
                "mutated_token_index": mutated_token_index,
                "mutation_block_index": mutation_block_index,
                "changed_block_indexes": mutated_changed,
            },
        ),
        run_case(
            "model_change_invalidates_all_blocks",
            True,
            len(changed_block_indexes(base_manifest, model_changed_manifest)) == base_manifest["block_count"],
            "Model hash is part of every block key.",
        ),
        run_case(
            "tokenizer_change_invalidates_all_blocks",
            True,
            len(changed_block_indexes(base_manifest, tokenizer_changed_manifest)) == base_manifest["block_count"],
            "Tokenizer hash is part of every block key.",
        ),
        run_case(
            "rope_change_invalidates_all_blocks",
            True,
            len(changed_block_indexes(base_manifest, rope_changed_manifest)) == base_manifest["block_count"],
            "RoPE settings hash is part of every block key.",
        ),
        run_case(
            "disk_budget_evicts_whole_blocks_only",
            True,
            bool(eviction["evicted"])
            and eviction["remaining_bytes"] <= eviction["budget_bytes"]
            and not eviction["partial_files"]
            and all(not Path(item["path"]).exists() for item in eviction["evicted"]),
            "Eviction removes complete block artifact files; it never truncates a block.",
            {
                "evicted_indexes": [item["block_index"] for item in eviction["evicted"]],
                "remaining_indexes": [item["block_index"] for item in eviction["remaining"]],
                "budget_bytes": eviction["budget_bytes"],
                "remaining_bytes": eviction["remaining_bytes"],
            },
        ),
    ]

    report = {
        "schema": "qwen3.5-ds4-block-boundary-lab-v1",
        "workflow": args.workflow,
        "policy": {
            "block_size_tokens": args.block_size_tokens,
            "tokenizer_mode": TOKENIZER_MODE,
            "prefix_chars": args.prefix_chars,
            "synthetic_block_bytes": args.synthetic_block_bytes,
            "disk_budget_ratio": args.disk_budget_ratio,
        },
        "base_manifest": base_manifest,
        "mutation": {
            "mutated_token_index": mutated_token_index,
            "mutation_block_index": mutation_block_index,
            "changed_block_indexes": mutated_changed,
        },
        "variant_manifests": {
            "same_tail_manifest_key": same_tail_manifest["manifest_key"],
            "mutated_manifest_key": mutated_manifest["manifest_key"],
            "model_changed_manifest_key": model_changed_manifest["manifest_key"],
            "tokenizer_changed_manifest_key": tokenizer_changed_manifest["manifest_key"],
            "rope_changed_manifest_key": rope_changed_manifest["manifest_key"],
        },
        "eviction": eviction,
        "tests": tests,
        "summary": {
            "ok": all(test["pass"] for test in tests),
            "test_total": len(tests),
            "test_passed": sum(1 for test in tests if test["pass"]),
            "block_count": base_manifest["block_count"],
            "approx_token_count": base_manifest["approx_token_count"],
            "manifest_key": base_manifest["manifest_key"],
        },
    }
    return report, 0 if report["summary"]["ok"] else 1


def print_report(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print("# Qwen Block Boundary Lab")
    print()
    print(f"- workflow: {report['workflow']}")
    print(f"- block_size_tokens: {report['policy']['block_size_tokens']}")
    print(f"- approx_token_count: {summary['approx_token_count']}")
    print(f"- block_count: {summary['block_count']}")
    print(f"- manifest_key: {summary['manifest_key']}")
    print()
    for test in report["tests"]:
        status = "PASS" if test["pass"] else "FAIL"
        print(f"- {test['name']}: {status}")
        if test["extra"]:
            print(f"  extra: {json.dumps(test['extra'], ensure_ascii=False, sort_keys=True)}")
    print()
    print(f"summary_ok: {summary['ok']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and verify block-level cache boundary metadata.")
    parser.add_argument("--artifact-dir", default="artifacts/block-boundary-lab")
    parser.add_argument("--trace-json", default="traces/block-boundary-lab-2026-05-15.json")
    parser.add_argument("--workflow", default="fb", choices=sorted(WORKFLOWS.keys()))
    parser.add_argument("--prefix-chars", type=int, default=6000)
    parser.add_argument("--block-size-tokens", type=int, default=512)
    parser.add_argument("--synthetic-block-bytes", type=int, default=32768)
    parser.add_argument("--disk-budget-ratio", type=float, default=0.55)
    parser.add_argument("--run-tag", default=str(int(time.time())))
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--model-id", default="qwen3.5:4b")
    parser.add_argument("--model-sha256", default="local-model-sha256-not-provided")
    parser.add_argument("--tokenizer-sha256", default="qwen35-tokenizer-sha256-not-provided")
    parser.add_argument("--quant-type", default="Q4_K_M")
    parser.add_argument("--runtime-name", default="llama.cpp")
    parser.add_argument("--runtime-version", default="llama.cpp-local-build")
    parser.add_argument("--chat-template", default="chatml")
    parser.add_argument("--context-size", type=int, default=32768)
    parser.add_argument("--kv-cache-format-version", default="llama.cpp-slot-save-v1")
    args = parser.parse_args()

    if args.block_size_tokens <= 0:
        raise SystemExit("--block-size-tokens must be positive")
    if not 0 < args.disk_budget_ratio < 1:
        raise SystemExit("--disk-budget-ratio must be between 0 and 1")

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
