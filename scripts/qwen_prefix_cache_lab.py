#!/usr/bin/env python3
"""
Measure fixed-prefix cache behavior for a Qwen3.5 OpenAI-compatible worker.

This is a practical harness for the DS4-style question:

    Can we avoid paying the prefill cost again when a long, stable prompt
    prefix is reused?

It does not implement SSD KV cache by itself. It measures what the current
runtime does, and it makes cache-key hazards visible by varying the stable
prefix and a pseudo steering-state marker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


BASE_BLOCK = """
固定任務背景：
- 你是本地 Qwen3.5 worker，用於小企業 AI 自動化。
- 回覆必須使用繁體中文。
- 目標是把長 SOP、技能規則、品牌語氣、品質管理規則先固定住。
- 使用情境包含：日本熱門話題改寫、社群內容再利用、rooming list OCR 補強、n8n SOP 檢查。
- 品質規則：不要杜撰來源、不要把連結放進正式 FB 文案、圖片版權不明時改做原創圖卡。
- 輸出要短、準、可人工審稿。
- 對敏感事實要標示需要查證。
"""


@dataclass
class ProbeCase:
    case_id: str
    group: str
    system_prefix: str
    user_tail: str
    steering_state: str


@dataclass
class ProbeResult:
    case_id: str
    group: str
    ok: bool
    prompt_sha1: str
    steering_state: str
    prompt_chars: int
    ttft_s: float | None
    latency_s: float
    usage: dict[str, Any] | None
    cached_tokens: int | None
    raw_preview: str
    error: str | None


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def make_prefix(target_chars: int, steering_state: str = "none", mutate_middle: bool = False) -> str:
    header = (
        "SYSTEM CACHE TEST HEADER\n"
        f"PSEUDO_STEERING_STATE={steering_state}\n"
        "If this marker changes, a safe cache key must treat the prefix as different.\n"
        "\n"
    )
    blocks: list[str] = []
    while len(header) + sum(len(item) for item in blocks) < target_chars:
        idx = len(blocks) + 1
        block = BASE_BLOCK.replace("固定任務背景", f"固定任務背景 #{idx:03d}")
        blocks.append(block)
    if mutate_middle and blocks:
        mid = len(blocks) // 2
        blocks[mid] = blocks[mid].replace("不要杜撰來源", "來源必須二次查證，且未確認時不可發布")
    return (header + "".join(blocks))[:target_chars]


def build_cases(prefix_chars: int) -> list[ProbeCase]:
    base = make_prefix(prefix_chars, steering_state="none")
    mutated = make_prefix(prefix_chars, steering_state="none", mutate_middle=True)
    alpha = make_prefix(prefix_chars, steering_state="alpha")
    beta = make_prefix(prefix_chars, steering_state="beta")
    question_a = (
        "請根據固定背景，輸出 JSON："
        "{\"recommendation\":\"...\",\"risk\":\"...\",\"next_step\":\"...\"}。"
        "主題：今天日本熱門話題改寫成 FB 文案前，最重要的 QC 是什麼？/no_think"
    )
    question_b = (
        "請根據固定背景，輸出 JSON："
        "{\"recommendation\":\"...\",\"risk\":\"...\",\"next_step\":\"...\"}。"
        "主題：n8n 自動化接 AI 之前，最該設定的成本邊界是什麼？/no_think"
    )
    return [
        ProbeCase("cold_fixed_a", "fixed-repeat", base, question_a, "none"),
        ProbeCase("warm_fixed_a_1", "fixed-repeat", base, question_a, "none"),
        ProbeCase("warm_fixed_a_2", "fixed-repeat", base, question_a, "none"),
        ProbeCase("same_prefix_new_tail", "same-prefix-new-tail", base, question_b, "none"),
        ProbeCase("mutated_middle", "prefix-mutation", mutated, question_a, "none"),
        ProbeCase("pseudo_steering_alpha_1", "pseudo-steering", alpha, question_a, "alpha"),
        ProbeCase("pseudo_steering_alpha_2", "pseudo-steering", alpha, question_a, "alpha"),
        ProbeCase("pseudo_steering_beta", "pseudo-steering", beta, question_a, "beta"),
    ]


def post_json(url: str, payload: dict[str, Any], timeout: float) -> tuple[dict[str, Any], float]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer qwen-prefix-cache-lab"},
        method="POST",
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    return json.loads(data.decode("utf-8")), time.perf_counter() - start


def iter_sse(url: str, payload: dict[str, Any], timeout: float):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer qwen-prefix-cache-lab"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or line.startswith(":") or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            yield json.loads(data)


def cached_tokens_from_usage(usage: dict[str, Any] | None) -> int | None:
    if not usage:
        return None
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict):
        cached = details.get("cached_tokens")
        if isinstance(cached, int):
            return cached
    cached = usage.get("cached_tokens")
    if isinstance(cached, int):
        return cached
    return None


def build_payload(model: str, case: ProbeCase, max_tokens: int, stream: bool) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": case.system_prefix
                + "\n\n回覆限制：只輸出一個 JSON object，不要 markdown，不要解釋。",
            },
            {"role": "user", "content": case.user_tail},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": stream,
        "stream_options": {"include_usage": True},
        "response_format": {"type": "json_object"},
        "think": False,
        "reasoning_effort": "none",
    }


def run_case(base_url: str, model: str, case: ProbeCase, max_tokens: int, stream: bool, timeout: float) -> ProbeResult:
    prompt_text = case.system_prefix + "\n" + case.user_tail
    payload = build_payload(model, case, max_tokens, stream)
    start = time.perf_counter()
    try:
        usage: dict[str, Any] | None = None
        if stream:
            first_at: float | None = None
            text_parts: list[str] = []
            for event in iter_sse(f"{base_url}/chat/completions", payload, timeout):
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    delta = choice.get("delta") or {}
                    content = delta.get("content")
                    if isinstance(content, str) and content:
                        if first_at is None:
                            first_at = time.perf_counter()
                        text_parts.append(content)
            raw = "".join(text_parts)
            ttft = None if first_at is None else first_at - start
            latency = time.perf_counter() - start
        else:
            data, latency = post_json(f"{base_url}/chat/completions", payload, timeout)
            usage = data.get("usage")
            choices = data.get("choices") or []
            message = choices[0].get("message", {}) if choices else {}
            raw = str(message.get("content") or "")
            ttft = None
        return ProbeResult(
            case_id=case.case_id,
            group=case.group,
            ok=bool(raw.strip()),
            prompt_sha1=sha1_text(prompt_text),
            steering_state=case.steering_state,
            prompt_chars=len(prompt_text),
            ttft_s=ttft,
            latency_s=latency,
            usage=usage,
            cached_tokens=cached_tokens_from_usage(usage),
            raw_preview=raw[:300].replace("\n", "\\n"),
            error=None,
        )
    except Exception as exc:
        return ProbeResult(
            case_id=case.case_id,
            group=case.group,
            ok=False,
            prompt_sha1=sha1_text(prompt_text),
            steering_state=case.steering_state,
            prompt_chars=len(prompt_text),
            ttft_s=None,
            latency_s=time.perf_counter() - start,
            usage=None,
            cached_tokens=None,
            raw_preview="",
            error=repr(exc),
        )


def health_check(base_url: str, timeout: float) -> dict[str, Any]:
    data, _ = post_json(
        f"{base_url}/chat/completions",
        {
            "model": "__health_check__",
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
        },
        timeout,
    )
    return data


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    idx = min(len(values) - 1, int(round((len(values) - 1) * pct)))
    return values[idx]


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def print_summary(results: list[ProbeResult]) -> None:
    print("# Qwen Prefix Cache Lab")
    print()
    for result in results:
        cached = "n/a" if result.cached_tokens is None else str(result.cached_tokens)
        ttft = fmt(result.ttft_s)
        print(
            f"- {result.case_id}: ok={result.ok} group={result.group} "
            f"ttft_s={ttft} latency_s={result.latency_s:.3f} "
            f"cached_tokens={cached} sha1={result.prompt_sha1[:10]} "
            f"steering={result.steering_state}"
        )
        if result.error:
            print(f"  error={result.error}")
    print()

    first = next((item for item in results if item.case_id == "cold_fixed_a"), None)
    warm = [item for item in results if item.case_id.startswith("warm_fixed")]
    warm_ttfts = [item.ttft_s for item in warm if item.ttft_s is not None]
    if first and first.ttft_s is not None and warm_ttfts:
        warm_p50 = percentile(warm_ttfts, 0.5)
        speedup = None if not warm_p50 else first.ttft_s / warm_p50
        print("## Fixed Prefix Repeat")
        print(f"- cold_ttft_s: {first.ttft_s:.3f}")
        print(f"- warm_ttft_p50_s: {fmt(warm_p50)}")
        print(f"- apparent_ttft_speedup: {fmt(speedup)}x")
        print()

    by_group: dict[str, list[float]] = {}
    for item in results:
        if item.ttft_s is not None:
            by_group.setdefault(item.group, []).append(item.ttft_s)
    if by_group:
        print("## Group TTFT")
        for group, values in by_group.items():
            print(f"- {group}: avg={statistics.mean(values):.3f} p50={fmt(percentile(values, 0.5))}")
        print()

    print("## Read This")
    print("- A large cold->warm TTFT drop means the current runtime is reusing a fixed prefix in RAM/session cache.")
    print("- If cached_tokens appears, trust that over timing; timing can be noisy.")
    print("- The pseudo-steering cases only simulate cache-key separation with visible text.")
    print("- Real vector steering needs the hidden steering state included in the KV cache metadata/hash.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure Qwen3.5 fixed-prefix cache behavior.")
    parser.add_argument("--base-url", default="http://127.0.0.1:18080/v1")
    parser.add_argument("--model", default="qwen3.5:4b")
    parser.add_argument("--prefix-chars", type=int, default=12000)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--trace-jsonl", default="traces-qwen35-prefix-cache-lab.jsonl")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    stream = not args.no_stream
    try:
        cases = build_cases(args.prefix_chars)
        results = [
            run_case(base_url, args.model, case, args.max_tokens, stream, args.timeout)
            for case in cases
        ]
    except urllib.error.URLError as exc:
        print(f"Could not reach {base_url}: {exc}", file=sys.stderr)
        return 2

    print_summary(results)
    if args.trace_jsonl:
        with open(args.trace_jsonl, "w", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(result.__dict__, ensure_ascii=False, default=str) + "\n")
        print(f"trace_jsonl: {args.trace_jsonl}")
    return 0 if all(item.ok for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
