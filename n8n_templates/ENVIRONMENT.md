# n8n Environment Variables

這份是給 n8n / Zeabur 設定用的環境變數清單。

範本檔：

```text
n8n_templates/env.example
```

請把值複製到 n8n 的 environment variables，不要把真正 API key commit 進 repo。

## Minimum For Local Test

只測本地 Qwen：

```text
LOCAL_AI_BASE_URL=http://host.docker.internal:18180/v1/chat/completions
LOCAL_AI_MODEL=qwen3.5:4b
LOCAL_AI_INPUT_USD_PER_1K=0
LOCAL_AI_OUTPUT_USD_PER_1K=0
REVIEW_SHEET_ID=<your-google-sheet-id>
TIMEZONE=Asia/Taipei
```

如果 Zeabur n8n 連不到你的 Mac，本地 URL 要換成 tunnel 或 private network URL。

## Minimum For Local vs Cloud Benchmark

```text
LOCAL_AI_BASE_URL=http://host.docker.internal:18180/v1/chat/completions
LOCAL_AI_MODEL=qwen3.5:4b
LOCAL_AI_INPUT_USD_PER_1K=0
LOCAL_AI_OUTPUT_USD_PER_1K=0

CLOUD_AI_BASE_URL=<openai-compatible-chat-completions-url>
CLOUD_AI_MODEL=<cloud-model-name>
CLOUD_AI_MODEL_ALT=<second-cloud-model-name>
CLOUD_AI_API_KEY=<cloud-api-key>
CLOUD_AI_INPUT_USD_PER_1K=<input-price>
CLOUD_AI_OUTPUT_USD_PER_1K=<output-price>
CLOUD_AI_ALT_INPUT_USD_PER_1K=<optional-alt-input-price>
CLOUD_AI_ALT_OUTPUT_USD_PER_1K=<optional-alt-output-price>

REVIEW_SHEET_ID=<your-google-sheet-id>
TIMEZONE=Asia/Taipei
```

## Gemini Example

如果你用 Gemini 的 OpenAI-compatible endpoint：

```text
CLOUD_AI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/chat/completions
CLOUD_AI_MODEL=gemini-3.1-flash-lite
CLOUD_AI_MODEL_ALT=gemma-4-26b-a4b-it
CLOUD_AI_API_KEY=<your-gemini-api-key>
CLOUD_AI_INPUT_USD_PER_1K=<current-input-price>
CLOUD_AI_OUTPUT_USD_PER_1K=<current-output-price>
CLOUD_AI_ALT_INPUT_USD_PER_1K=<current-alt-input-price>
CLOUD_AI_ALT_OUTPUT_USD_PER_1K=<current-alt-output-price>
```

價格請用你當下 API 後台或官方價格頁填入。

## OpenAI Example

```text
CLOUD_AI_BASE_URL=https://api.openai.com/v1/chat/completions
CLOUD_AI_MODEL=<model-name>
CLOUD_AI_API_KEY=<your-openai-api-key>
CLOUD_AI_INPUT_USD_PER_1K=<current-input-price>
CLOUD_AI_OUTPUT_USD_PER_1K=<current-output-price>
```

## Google Sheet Tabs

建議先建立這些分頁：

```text
AI Benchmarks
Content Review
Leads
Support Review
Daily Digest
AI Trend Review
Review Replies
```

benchmark workflow 主要用：

```text
AI Benchmarks
```

## Cost Estimate

模板用這個公式估算：

```text
input_tokens / 1000 * INPUT_USD_PER_1K
+ output_tokens / 1000 * OUTPUT_USD_PER_1K
```

本地預設為 0。若要算電費或硬體攤提，可以自己填入估算值。
