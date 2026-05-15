#!/usr/bin/env python3
"""
Benchmark translation workloads with and without reusable fixed-prefix SOP.

The goal is practical: if a solo founder or small team translates many similar
articles, does a cached long SOP/glossary beat repeatedly sending a changing
long prompt?

This script uses synthetic article segments to avoid copyright issues.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qwen_prefix_cache_lab import cached_tokens_from_usage, iter_sse, percentile, post_json


SOURCE_SEGMENTS = [
    (
        "日本の小規模事業者の間で、AIを使った業務自動化への関心が高まっている。"
        "特に、問い合わせ対応、商品説明の作成、社内マニュアルの検索、SNS投稿の再利用など、"
        "毎日繰り返される作業から導入する例が増えている。"
    ),
    (
        "一方で、AIエージェントを導入する企業からは、利用料金の管理が難しいという声もある。"
        "自動化された処理が長時間ループすると、想定以上のトークンを消費する可能性があるため、"
        "上限設定とログ確認が重要になる。"
    ),
    (
        "観光業界では、訪日客向けにメニュー翻訳、レシート解析、旅程整理を支援するツールが注目されている。"
        "ただし、翻訳結果をそのまま店舗スタッフに見せるだけでは、売り切れやセット内容の確認に対応できない場合がある。"
    ),
    (
        "小売店では、商品ページの改善にAIを使う事例が広がっている。"
        "写真の説明、検索キーワード、レビュー要約を組み合わせることで、少人数でも継続的にページを更新しやすくなる。"
    ),
    (
        "ローカルLLMを活用する開発者は、固定されたシステムプロンプトや業務ルールを再利用することで、"
        "初回の読み込み時間を短縮できるかに注目している。長い前提条件を毎回処理する必要がなくなれば、"
        "体感速度は大きく変わる。"
    ),
    (
        "ただし、キャッシュを安全に使うには、モデルの種類、量子化方式、トークナイザー、"
        "コンテキスト長、追加の制御ベクトルなどを区別する必要がある。異なる状態のKVキャッシュを誤って使うと、"
        "出力品質が不安定になる。"
    ),
    (
        "コンテンツ制作では、一つの長い記事をThreads、Instagram、Facebook、メールマガジン向けに再利用する流れが一般的になりつつある。"
        "AIは下書きを素早く作れるが、最終的な事実確認とブランド表現の調整は人間が行う必要がある。"
    ),
    (
        "今後は、クラウドAPIとローカルモデルを使い分ける構成が増えると考えられる。"
        "機密性の高い文書や頻繁に使う社内知識はローカルで処理し、最新情報の確認や高難度の推論はクラウドに任せる形だ。"
    ),
]


LONG_TRANSLATION_SOP = """
TRANSLATION SOP FOR SMALL BUSINESS CONTENT WORKER

Role:
- You are a local translation worker for Taiwanese solo founders and small teams.
- Translate Japanese business / travel / AI operations articles into Traditional Chinese.
- Output must be natural Taiwan Mandarin, not literal machine translation.

Quality rules:
- Preserve factual uncertainty. Do not add claims that are not in the source.
- Keep brand/product/tool names as-is unless a common Traditional Chinese name exists.
- Translate "AIエージェント" as "AI Agent" unless the sentence needs plain-language explanation.
- Translate "ローカルLLM" as "本地 LLM".
- Translate "問い合わせ対応" as "客服回覆" or "客服處理" depending on context.
- Translate "業務自動化" as "工作流程自動化" when speaking to small businesses.
- Prefer short sentences suitable for Facebook and Threads repurposing.
- Avoid Mainland Chinese terms when Taiwan usage is more natural.
- Do not include markdown tables.
- Do not summarize unless explicitly asked.
- Do not leave Japanese punctuation style when Chinese punctuation is clearer.
- Use Arabic numerals when the source uses quantities.
- If a sentence contains risk, cost, compliance, or privacy implications, make it clear.

Terminology glossary:
- 小規模事業者: 小型業者 / 小企業經營者
- 業務自動化: 工作流程自動化
- 問い合わせ対応: 客服回覆
- 社内マニュアル: 內部手冊
- SNS投稿: 社群貼文
- 利用料金: 使用費
- トークン: token
- 上限設定: 用量上限
- ログ確認: log 檢查
- 訪日客: 訪日旅客
- メニュー翻訳: 菜單翻譯
- レシート解析: 收據解析
- 旅程整理: 行程整理
- 店舗スタッフ: 店員
- 商品ページ: 商品頁
- 検索キーワード: 搜尋關鍵字
- 量子化方式: 量化方式
- 制御ベクトル: 控制向量
- 出力品質: 輸出品質
- 下書き: 草稿
- 事実確認: 事實查證
- 機密性の高い文書: 高機密文件

Style examples:
- Bad: 「AI的導入變得高漲」
- Good: 「越來越多小企業開始導入 AI」
- Bad: 「使用量的管理很困難」
- Good: 「最麻煩的是用量不好控」
- Bad: 「對應客人的詢問」
- Good: 「處理客戶詢問」

Output contract:
- Return only the translated Traditional Chinese paragraph.
- No explanation.
- No title unless the source has one.
- No source URL.
"""


MINIMAL_TRANSLATION_SOP = "Translate the Japanese text into natural Traditional Chinese. Output only the translation."


@dataclass
class BenchResult:
    group: str
    index: int
    ok: bool
    prompt_chars: int
    source_chars: int
    ttft_s: float | None
    latency_s: float
    usage: dict[str, Any] | None
    cached_tokens: int | None
    output_chars: int
    raw_preview: str
    error: str | None


def expand_sop(target_chars: int, marker: str = "") -> str:
    header = ""
    if marker:
        header = f"RUN_MARKER={marker}\n"
    blocks: list[str] = []
    while len(header) + sum(len(block) for block in blocks) < target_chars:
        blocks.append(LONG_TRANSLATION_SOP)
    return (header + "".join(blocks))[:target_chars]


def build_payload(model: str, system_prompt: str, source_text: str, max_tokens: int, stream: bool) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"請翻譯以下日文段落：\n\n{source_text}\n/no_think"},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": stream,
        "stream_options": {"include_usage": True},
        "think": False,
        "reasoning_effort": "none",
    }


def run_one(
    *,
    base_url: str,
    model: str,
    group: str,
    index: int,
    system_prompt: str,
    source_text: str,
    max_tokens: int,
    stream: bool,
    timeout: float,
) -> BenchResult:
    payload = build_payload(model, system_prompt, source_text, max_tokens, stream)
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
        return BenchResult(
            group=group,
            index=index,
            ok=bool(raw.strip()),
            prompt_chars=len(system_prompt) + len(source_text),
            source_chars=len(source_text),
            ttft_s=ttft,
            latency_s=latency,
            usage=usage,
            cached_tokens=cached_tokens_from_usage(usage),
            output_chars=len(raw),
            raw_preview=raw[:220].replace("\n", "\\n"),
            error=None,
        )
    except Exception as exc:
        return BenchResult(
            group=group,
            index=index,
            ok=False,
            prompt_chars=len(system_prompt) + len(source_text),
            source_chars=len(source_text),
            ttft_s=None,
            latency_s=time.perf_counter() - start,
            usage=None,
            cached_tokens=None,
            output_chars=0,
            raw_preview="",
            error=repr(exc),
        )


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def summarize(results: list[BenchResult]) -> dict[str, Any]:
    groups = sorted({item.group for item in results})
    summary: dict[str, Any] = {}
    for group in groups:
        rows = [item for item in results if item.group == group]
        latencies = [item.latency_s for item in rows if item.ok]
        ttfts = [item.ttft_s for item in rows if item.ttft_s is not None and item.ok]
        cached = [item.cached_tokens for item in rows if item.cached_tokens is not None]
        warm_rows = rows[1:] if len(rows) > 1 else rows
        warm_ttfts = [item.ttft_s for item in warm_rows if item.ttft_s is not None and item.ok]
        warm_latencies = [item.latency_s for item in warm_rows if item.ok]
        summary[group] = {
            "total": len(rows),
            "ok": sum(1 for item in rows if item.ok),
            "ttft_avg_s": statistics.mean(ttfts) if ttfts else None,
            "ttft_p50_s": percentile(ttfts, 0.50) if ttfts else None,
            "latency_avg_s": statistics.mean(latencies) if latencies else None,
            "latency_p50_s": percentile(latencies, 0.50) if latencies else None,
            "warm_ttft_avg_s": statistics.mean(warm_ttfts) if warm_ttfts else None,
            "warm_latency_avg_s": statistics.mean(warm_latencies) if warm_latencies else None,
            "cached_tokens_avg": statistics.mean(cached) if cached else None,
            "cached_tokens_max": max(cached) if cached else None,
        }
    return summary


def print_report(results: list[BenchResult], summary: dict[str, Any]) -> None:
    print("# Qwen Translation Cache Bench")
    print()
    for item in results:
        cached = "n/a" if item.cached_tokens is None else str(item.cached_tokens)
        print(
            f"- {item.group}[{item.index}]: ok={item.ok} "
            f"ttft_s={fmt(item.ttft_s)} latency_s={item.latency_s:.3f} "
            f"cached_tokens={cached} output_chars={item.output_chars}"
        )
        if item.error:
            print(f"  error={item.error}")
    print()
    print("## Summary")
    for group, row in summary.items():
        print(
            f"- {group}: ok={row['ok']}/{row['total']} "
            f"ttft_avg={fmt(row['ttft_avg_s'])} latency_avg={fmt(row['latency_avg_s'])} "
            f"warm_ttft_avg={fmt(row['warm_ttft_avg_s'])} warm_latency_avg={fmt(row['warm_latency_avg_s'])} "
            f"cached_avg={fmt(row['cached_tokens_avg'])} cached_max={row['cached_tokens_max']}"
        )
    if "cached_long_sop" in summary and "busted_long_sop" in summary:
        cached = summary["cached_long_sop"].get("warm_ttft_avg_s")
        busted = summary["busted_long_sop"].get("warm_ttft_avg_s")
        if cached and busted:
            print()
            print("## Warm TTFT Delta")
            print(f"- busted_long_sop / cached_long_sop: {busted / cached:.2f}x")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark translation speed with reusable vs busted long SOP.")
    parser.add_argument("--base-url", default="http://127.0.0.1:18180/v1")
    parser.add_argument("--model", default="qwen3.5:4b")
    parser.add_argument("--segments", type=int, default=6)
    parser.add_argument("--sop-chars", type=int, default=6000)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--trace-jsonl", default="traces/translation-cache-bench.jsonl")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    stream = not args.no_stream
    segments = [SOURCE_SEGMENTS[i % len(SOURCE_SEGMENTS)] for i in range(args.segments)]
    stable_sop = expand_sop(args.sop_chars)
    results: list[BenchResult] = []

    for idx, segment in enumerate(segments):
        results.append(
            run_one(
                base_url=base_url,
                model=args.model,
                group="cached_long_sop",
                index=idx,
                system_prompt=stable_sop,
                source_text=segment,
                max_tokens=args.max_tokens,
                stream=stream,
                timeout=args.timeout,
            )
        )

    for idx, segment in enumerate(segments):
        results.append(
            run_one(
                base_url=base_url,
                model=args.model,
                group="busted_long_sop",
                index=idx,
                system_prompt=expand_sop(args.sop_chars, marker=f"segment-{idx}-{time.time_ns()}"),
                source_text=segment,
                max_tokens=args.max_tokens,
                stream=stream,
                timeout=args.timeout,
            )
        )

    for idx, segment in enumerate(segments):
        results.append(
            run_one(
                base_url=base_url,
                model=args.model,
                group="minimal_sop",
                index=idx,
                system_prompt=MINIMAL_TRANSLATION_SOP,
                source_text=segment,
                max_tokens=args.max_tokens,
                stream=stream,
                timeout=args.timeout,
            )
        )

    summary = summarize(results)
    print_report(results, summary)
    if args.trace_jsonl:
        path = Path(args.trace_jsonl)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for item in results:
                handle.write(json.dumps(item.__dict__, ensure_ascii=False, default=str) + "\n")
        summary_path = path.with_suffix(".summary.json")
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print()
        print(f"trace_jsonl: {path}")
        print(f"summary_json: {summary_path}")
    return 0 if all(item.ok for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
