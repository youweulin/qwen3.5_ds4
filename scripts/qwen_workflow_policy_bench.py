#!/usr/bin/env python3
"""
Benchmark one cache policy across several local-agent workflows.

This is intentionally about speed/cache behavior on Qwen3.5 4B, not final
content quality. The same external manager policy is applied to:

- FB content SOP
- translation glossary
- rooming-list QC

For each workflow:
1. baseline cold follow-up without manager
2. manager miss, run source task, save .qkv
3. erase slot
4. manager hit, restore .qkv, run follow-up task
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qwen_cache_key_lab import cache_key, make_metadata, sha256_bytes, sha256_text
from qwen_kv_artifact_lab import artifact_filename, read_artifact, verify_artifact, write_artifact


@dataclass(frozen=True)
class Workflow:
    workflow_id: str
    title: str
    base_sop: str
    seed_task: str
    followup_task: str
    output_contract: str


FB_BASE_SOP = """
你是台灣繁體中文社群內容改寫 worker，任務是把日本熱門話題改寫成 FB 貼文草稿。
固定規則：
- 不放 URL，不做連結預覽。
- 可以寫「出處：媒體或官方名稱」，但不要貼網址。
- 第一行要有 hook，避免像新聞稿。
- 要保留可查證事實，不可補不存在的數字、品牌、時間。
- 如果事實不確定，輸出 needs_fact_check=true。
- 語氣要像個人品牌觀察，不要像廣告或農場文。
- 避免過度煽動，避免直接攻擊個人。
- 輸出 JSON，欄位固定：hook, body, source_name, risk, needs_fact_check。
品質檢查：
- 是否有 URL？有就 fail。
- 是否有繁中口吻？沒有就 fail。
- 是否有清楚出處名稱？沒有就 warning。
- 是否把國際局勢或商業影響講成人話？沒有就 warning。
"""

TRANSLATION_BASE_SOP = """
你是日文到台灣繁體中文翻譯 worker，任務是大量翻譯商業/旅遊/科技文章段落。
固定術語：
- インバウンド -> 訪日旅客
- 生成AI -> 生成式 AI
- 事業者 -> 業者
- 予約 -> 預約
- 決済 -> 支付
- 導入 -> 導入
- 自治体 -> 地方政府
- 見直し -> 重新檢討
固定規則：
- 使用台灣常見說法，不使用中國大陸用語。
- 保留原意，不自行補背景。
- 專有名詞第一次出現保留原文括號。
- 數字、日期、金額不可改。
- 輸出 JSON，欄位固定：translation, glossary_hits, risk, needs_review。
品質檢查：
- 術語是否一致。
- 有沒有漏翻否定句。
- 有沒有把推測寫成事實。
- 是否保留段落邏輯。
"""

ROOMING_BASE_SOP = """
你是旅遊團 rooming list QC worker，任務是檢查房號、人數、中文/英文姓名與備註。
固定規則：
- 不可以猜護照姓名，不確定就標 needs_human_check。
- 雙人房應有 2 人，三人房應有 3 人，單人房應有 1 人。
- 小孩、嬰兒、特殊餐、輪椅、導遊、領隊都要保留備註。
- 中文姓名和英文姓名要能對齊；明顯缺漏要列出。
- 若房型與人數不一致，列為 blocking_issue。
- 輸出 JSON，欄位固定：summary, blocking_issues, warnings, normalized_rooms。
品質檢查：
- 是否每間房都有房號。
- 是否每位旅客都有中文姓名與英文姓名。
- 是否標示需要人工確認的欄位。
- 是否避免自行補資料。
"""


WORKFLOWS = {
    "fb": Workflow(
        workflow_id="fb",
        title="FB content SOP",
        base_sop=FB_BASE_SOP,
        seed_task=(
            "日本熱門話題：Calbee 因石腦油與印刷油墨供應影響，14 款商品暫時改成黑白包裝。"
            "請改寫成 FB 貼文草稿。出處：日本主流媒體。"
        ),
        followup_task=(
            "日本熱門話題：地方鐵道因觀光客回流調整班次，但司機人力不足。"
            "請改寫成 FB 貼文草稿。出處：NHK。"
        ),
        output_contract="只輸出 JSON object，不要 markdown。",
    ),
    "translation": Workflow(
        workflow_id="translation",
        title="Translation glossary",
        base_sop=TRANSLATION_BASE_SOP,
        seed_task=(
            "次の文章を台湾繁体字中国語に翻訳してください："
            "地方自治体は訪日旅客の増加を受け、予約システムと決済手段の見直しを進めている。"
            "一部の事業者は生成AIを導入し、多言語対応の負担を減らそうとしている。"
        ),
        followup_task=(
            "次の文章を台湾繁体字中国語に翻訳してください："
            "観光施設では週末の混雑を避けるため、事前予約とキャッシュレス決済の導入が広がっている。"
            "自治体は小規模事業者への支援策も検討している。"
        ),
        output_contract="只輸出 JSON object，不要 markdown。",
    ),
    "rooming": Workflow(
        workflow_id="rooming",
        title="Rooming list QC",
        base_sop=ROOMING_BASE_SOP,
        seed_task=(
            "請檢查以下 rooming list：\n"
            "Room 201 TWN 張小明 CHANG/HSIAO MING; TWN 王美玲 WANG/MEI LING\n"
            "Room 202 TRP 陳大文 CHEN/TA WEN; 林佳怡 LIN/CHIA YI\n"
            "Room 203 SGL 領隊 李志強 LEE/CHIH CHIANG"
        ),
        followup_task=(
            "請檢查以下 rooming list：\n"
            "Room 301 TWN 黃俊傑 HUANG/CHUN CHIEH; 劉雅婷 LIU/YA TING\n"
            "Room 302 TWN 蔡佩君 TSAI/PEI CHUN\n"
            "Room 303 TRP 吳宗翰 WU/TSUNG HAN; 鄭怡如 CHENG/I JU; infant 王小寶"
        ),
        output_contract="只輸出 JSON object，不要 markdown。",
    ),
}


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


def repeat_to_chars(text: str, target_chars: int) -> str:
    chunks = [text.strip()]
    idx = 1
    while len("\n\n".join(chunks)) < target_chars:
        chunks.append(
            f"延伸規則 #{idx:03d}：沿用上述 SOP。遇到不確定資訊時，要標示 needs_review，不要自行補資料。"
        )
        idx += 1
    return "\n\n".join(chunks)[:target_chars]


def build_prompt(workflow: Workflow, prefix: str, task: str) -> str:
    return (
        f"{prefix}\n\n"
        f"輸出契約：{workflow.output_contract}\n"
        "請只處理 USER_TASK，不要引用本段系統文字。\n\n"
        f"USER_TASK:\n{task}\n/no_think\nASSISTANT:\n"
    )


def make_workflow_prefix(workflow: Workflow, target_chars: int, run_tag: str) -> str:
    prefix = (
        f"WORKFLOW_POLICY_BENCH_RUN={run_tag}\n"
        f"WORKFLOW_ID={workflow.workflow_id}\n"
        f"WORKFLOW_TITLE={workflow.title}\n"
        "目的：測速度與 cache policy，不用 4B 評估最終內容品質。\n\n"
        f"{workflow.base_sop}"
    )
    return repeat_to_chars(prefix, target_chars)


def make_prefix_metadata(args: argparse.Namespace, workflow: Workflow, prefix_text: str):
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
        policy_sha256=sha256_text(f"workflow-policy-v1:{workflow.workflow_id}"),
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
    restore_response = None
    if artifact_path.exists():
        _header, payload = read_artifact(artifact_path)
        verify_artifact(artifact_path, metadata)
        restore_filename = f"workflow-restore-{key}.bin"
        write_slot_payload(args.slot_save_path, restore_filename, payload)
        restore_response, restore_latency_s = slot_action(
            args.base_url,
            args.slot_id,
            "restore",
            args.timeout,
            {"filename": restore_filename},
        )
        restored = bool(restore_response.get("n_restored", 0) > 0)

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
        save_filename = f"workflow-save-{key}.bin"
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


def run_workflow(args: argparse.Namespace, workflow: Workflow) -> dict[str, Any]:
    baseline_prefix = make_workflow_prefix(workflow, args.prefix_chars, f"{args.run_tag}-baseline-{workflow.workflow_id}")
    baseline_prompt = build_prompt(workflow, baseline_prefix, workflow.followup_task)
    baseline_metadata = make_prefix_metadata(args, workflow, baseline_prefix)
    slot_action(args.base_url, args.slot_id, "erase", args.timeout)
    baseline_response, baseline_latency_s = completion(
        args.base_url,
        args.slot_id,
        baseline_prompt,
        args.n_predict,
        args.timeout,
    )
    baseline = {
        "prefix_cache_key": cache_key(baseline_metadata),
        "latency_s": baseline_latency_s,
        "prompt_n": timings_prompt_n(baseline_response),
        "completion_preview": str(baseline_response.get("content", ""))[:240],
    }

    managed_prefix = make_workflow_prefix(workflow, args.prefix_chars, f"{args.run_tag}-managed-{workflow.workflow_id}")
    managed_metadata = make_prefix_metadata(args, workflow, managed_prefix)
    seed_prompt = build_prompt(workflow, managed_prefix, workflow.seed_task)
    followup_prompt = build_prompt(workflow, managed_prefix, workflow.followup_task)

    slot_action(args.base_url, args.slot_id, "erase", args.timeout)
    manager_miss = managed_completion(args, seed_prompt, managed_metadata, allow_save_on_miss=True)
    slot_action(args.base_url, args.slot_id, "erase", args.timeout)
    manager_hit = managed_completion(args, followup_prompt, managed_metadata, allow_save_on_miss=False)

    hit_latency = manager_hit["total_latency_s"]
    speedup = baseline_latency_s / hit_latency if hit_latency else None
    prompt_n_saved = None
    if isinstance(baseline["prompt_n"], int) and isinstance(manager_hit["prompt_n"], int):
        prompt_n_saved = baseline["prompt_n"] - manager_hit["prompt_n"]

    return {
        "workflow_id": workflow.workflow_id,
        "title": workflow.title,
        "baseline_no_manager_cold_followup": baseline,
        "manager_miss_then_save": manager_miss,
        "manager_hit_restore_then_followup": manager_hit,
        "summary": {
            "ok": bool(manager_miss["cache_saved"] and manager_hit["cache_hit"]),
            "baseline_latency_s": baseline_latency_s,
            "manager_hit_total_latency_s": hit_latency,
            "manager_hit_restore_latency_s": manager_hit["restore_latency_s"],
            "manager_hit_completion_latency_s": manager_hit["completion_latency_s"],
            "latency_speedup_vs_cold": speedup,
            "baseline_prompt_n": baseline["prompt_n"],
            "manager_hit_prompt_n": manager_hit["prompt_n"],
            "prompt_n_saved": prompt_n_saved,
            "slot_payload_size": manager_miss["slot_payload_size"],
        },
    }


def run_lab(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    if args.clean:
        for path in [Path(args.artifact_dir), Path(args.slot_save_path)]:
            if path.exists():
                shutil.rmtree(path)
    Path(args.artifact_dir).mkdir(parents=True, exist_ok=True)
    Path(args.slot_save_path).mkdir(parents=True, exist_ok=True)

    selected = [item.strip() for item in args.workflows.split(",") if item.strip()]
    unknown = [item for item in selected if item not in WORKFLOWS]
    if unknown:
        raise SystemExit(f"unknown workflows: {', '.join(unknown)}")

    rows = [run_workflow(args, WORKFLOWS[item]) for item in selected]
    report = {
        "schema": "qwen3.5-ds4-workflow-policy-bench-v1",
        "base_url": args.base_url,
        "slot_id": args.slot_id,
        "policy": {
            "same_policy_for_all_workflows": True,
            "cache_scope": "prefix",
            "prefix_chars": args.prefix_chars,
            "n_predict": args.n_predict,
            "artifact_dir": args.artifact_dir,
            "slot_save_path": args.slot_save_path,
            "model_id": args.model_id,
            "quality_note": "Qwen3.5 4B run measures speed/cache only, not final workflow quality.",
        },
        "workflows": rows,
        "summary": {
            "total": len(rows),
            "passed": sum(1 for row in rows if row["summary"]["ok"]),
            "failed": sum(1 for row in rows if not row["summary"]["ok"]),
            "avg_speedup_vs_cold": (
                sum(row["summary"]["latency_speedup_vs_cold"] for row in rows if row["summary"]["latency_speedup_vs_cold"])
                / max(1, sum(1 for row in rows if row["summary"]["latency_speedup_vs_cold"]))
            ),
        },
    }
    return report, 0 if report["summary"]["failed"] == 0 else 1


def print_report(report: dict[str, Any]) -> None:
    print("# Qwen Workflow Policy Bench")
    print()
    for row in report["workflows"]:
        summary = row["summary"]
        speedup = summary["latency_speedup_vs_cold"]
        speedup_text = "n/a" if speedup is None else f"{speedup:.2f}x"
        print(f"## {row['workflow_id']} - {row['title']}")
        print(f"- baseline_latency_s: {summary['baseline_latency_s']:.3f}")
        print(f"- manager_hit_total_latency_s: {summary['manager_hit_total_latency_s']:.3f}")
        print(f"- manager_hit_restore_latency_s: {summary['manager_hit_restore_latency_s']:.3f}")
        print(f"- speedup_vs_cold: {speedup_text}")
        print(f"- baseline_prompt_n: {summary['baseline_prompt_n']}")
        print(f"- manager_hit_prompt_n: {summary['manager_hit_prompt_n']}")
        print(f"- prompt_n_saved: {summary['prompt_n_saved']}")
        print(f"- slot_payload_size: {summary['slot_payload_size']}")
        print()
    print("## Summary")
    print(f"- total: {report['summary']['total']}")
    print(f"- passed: {report['summary']['passed']}")
    print(f"- failed: {report['summary']['failed']}")
    print(f"- avg_speedup_vs_cold: {report['summary']['avg_speedup_vs_cold']:.2f}x")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark one cache policy across local-agent workflows.")
    parser.add_argument("--base-url", default="http://127.0.0.1:18180")
    parser.add_argument("--slot-id", type=int, default=0)
    parser.add_argument("--slot-save-path", default="artifacts/workflow-policy-slots/")
    parser.add_argument("--artifact-dir", default="artifacts/workflow-policy-artifacts")
    parser.add_argument("--trace-json", default="traces/workflow-policy-bench-2026-05-15.json")
    parser.add_argument("--workflows", default="fb,translation,rooming")
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
