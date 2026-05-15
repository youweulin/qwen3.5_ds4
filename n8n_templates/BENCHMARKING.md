# Benchmarking Local vs Cloud AI in n8n

目的：

```text
同一個 n8n 任務
同一份 policy
同一個 prompt
同一個 max_tokens
比較本地 AI 和雲端 AI 的速度與成本
```

## Workflow

Import:

```text
n8n_templates/workflows/ai-provider-speed-cost-benchmark.json
```

輸出會寫到 Google Sheet：

```text
AI Benchmarks
```

建議欄位：

```text
created_at
policy_name
prompt_chars
local_model
local_latency_ms
local_total_tokens
local_cost_usd
cloud_model
cloud_latency_ms
cloud_total_tokens
cloud_cost_usd
speed_ratio_cloud_over_local
cost_ratio_cloud_over_local
winner_speed
winner_cost
raw_json
```

## Required Environment Variables

本地：

```text
LOCAL_AI_BASE_URL=http://host.docker.internal:18180/v1/chat/completions
LOCAL_AI_MODEL=qwen3.5:4b
LOCAL_AI_INPUT_USD_PER_1K=0
LOCAL_AI_OUTPUT_USD_PER_1K=0
```

雲端：

```text
CLOUD_AI_BASE_URL=<openai-compatible-chat-completions-url>
CLOUD_AI_MODEL=<cloud-model-name>
CLOUD_AI_API_KEY=<api-key>
CLOUD_AI_INPUT_USD_PER_1K=<input-price>
CLOUD_AI_OUTPUT_USD_PER_1K=<output-price>
```

Google Sheet：

```text
REVIEW_SHEET_ID=<google-sheet-id>
```

## What To Measure

速度：

```text
local_latency_ms
cloud_latency_ms
speed_ratio_cloud_over_local = cloud_latency_ms / local_latency_ms
```

如果 ratio > 1：

```text
本地比較快
```

如果 ratio < 1：

```text
雲端比較快
```

成本：

```text
estimated_cost = input_tokens / 1000 * input_rate
               + output_tokens / 1000 * output_rate
```

本地成本預設是 0，因為硬體已經買了；如果要算電費或硬體攤提，可以把 `LOCAL_AI_*_USD_PER_1K` 填入自己的估算值。

## Recommended Test Sets

短任務：

```text
500-1000 tokens
客服回覆、短貼文、簡短摘要
```

中任務：

```text
3000-8000 tokens
文章改寫、翻譯、社群內容再利用
```

長固定 SOP 任務：

```text
10000-100000 tokens fixed policy/prefix
每次只換 tail
```

這是本地 prefix cache 最有機會贏的地方。

## Interpretation

雲端通常贏在：

```text
大模型品質
單次複雜推理
不用維護硬體
```

本地通常贏在：

```text
固定 SOP 重複工作
大量低風險任務
資料不想出門
沒有 API token 帳單壓力
可用 prefix cache 省 prefill
```

最實用的結論不是「本地或雲端誰取代誰」，而是 routing：

```text
本地：
  重複 SOP
  分類
  草稿
  QC
  摘要

雲端：
  高價值 final rewrite
  困難推理
  需要更強模型品質的任務
```

