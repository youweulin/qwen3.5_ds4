# n8n Templates for Solo / Small Business AI Workflows

這個資料區放的是「個人創業最需要」的 n8n workflow 草稿與 AI policy。

設計目標：

- 低成本：先用 Google Sheet / Gmail / Webhook / 本地 AI server。
- 可審稿：AI 產出先進 review queue，不直接亂發。
- 可快取：固定 SOP / 品牌語氣 / QC 規則放在 `policies/`，適合給 Qwen3.5 DS4-style prefix cache 使用。
- 可替換：HTTP Request 預設指向本地 OpenAI-compatible endpoint，也可以換成 Gemini / OpenAI / Dify / 自架 API。

## Import

1. 開 n8n。
2. 選 `Import from File`。
3. 匯入 `workflows/*.json`。
4. 檢查每個節點的 credential 與環境變數。
5. 先手動測試，再開啟 trigger。

## Recommended Environment Variables

```text
LOCAL_AI_BASE_URL=http://host.docker.internal:18180/v1/chat/completions
LOCAL_AI_MODEL=qwen3.5:4b
REVIEW_SHEET_ID=<google-sheet-id>
BRAND_NAME=<your-brand>
TIMEZONE=Asia/Taipei
```

如果 n8n 跑在 Zeabur / Docker，而 AI server 跑在本機，`host.docker.internal` 不一定可用。那時候要改成你的 tunnel / private network URL。

## Template Catalog

| Workflow | Best for | Human review | AI cache value |
| --- | --- | --- | --- |
| `content-repurpose-review.json` | 一篇文章/逐字稿產生 FB/IG/Threads/Email | Yes | High |
| `lead-intake-crm-followup.json` | 表單名單整理、分級、寄追蹤信 | Yes | Medium |
| `customer-support-triage.json` | 客服信件分類、草稿回覆、升級提醒 | Yes | High |
| `invoice-order-digest.json` | 訂單/發票整理、出貨摘要、每日報表 | Yes | Medium |
| `ai-trend-to-business-post.json` | AI 新技術整理成商業貼文 | Yes | High |
| `review-monitor-reply-draft.json` | Google 評論/社群留言回覆草稿 | Yes | High |

## AI Engine Pattern

固定 prefix：

```text
policy file
brand voice
output schema
quality checklist
forbidden actions
```

動態 tail：

```text
webhook input
email body
article transcript
lead form
order rows
```

這剛好適合目前引擎：

```text
第一次 cold prefill policy
save_prefix
之後每筆資料 restore_prefix + 只跑 tail
```

## Safety Defaults

- 不直接 publish。
- 不直接付款、退款、刪單、改訂單。
- 不把客戶個資送到未知第三方 API。
- 所有對外文案先進 Google Sheet review queue。
- AI output 要求 JSON，方便 n8n 後續節點處理。

