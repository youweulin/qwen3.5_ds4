# Qwen3.5 DS4-Style Prefix Cache Lab

這個 repo 記錄一個很小但很重要的本地 LLM 實驗：

> Qwen3.5 本地模型能不能像 DS4 的想法一樣，把固定長上下文算過的狀態重複利用，讓 Agent / RAG / SOP worker 變快？

目前結論：**可以吃到很明顯的固定 prefix prompt cache / checkpoint 效果**。  
但這還不是完整 DS4 那種「SSD KV persistence across clean restart」，目前比較接近 **RAM/session prompt cache + context checkpoint**。

## 背景

Antirez 在 DS4 開發中提到一個關鍵方向：AI 太重要，不能只是一個雲端服務。  
對小企業或個人工作流來說，本地模型真正有價值的地方不是「一次回答最強」，而是：

- 固定 SOP
- 固定品牌語氣
- 固定工具 schema
- 固定品質管理規則
- 固定 RAG 熱門文件

這些內容如果每次都重新 prefill，會浪費很多時間。  
如果能把固定 prefix 算過的狀態重用，就能把本地 Agent 的等待時間大幅壓低。

## 本次實驗目標

測試 Qwen3.5 4B 在 llama.cpp worker 路線下，對「長固定 prefix」是否有實際 cache 效果。

測試項目：

| Case | 意義 |
| --- | --- |
| `cold_fixed_a` | 第一次送長 prefix，預期最慢 |
| `warm_fixed_a_1` | 同一 prefix + 同一 user tail，測完整命中 |
| `warm_fixed_a_2` | 再重複一次，確認不是偶然 |
| `same_prefix_new_tail` | 同一 prefix，但換問題，測部分命中 |
| `mutated_middle` | prefix 中間改掉，應該接近 cold |
| `pseudo_steering_alpha_1` | 模擬 steering 狀態 alpha 的第一次 |
| `pseudo_steering_alpha_2` | 同一 steering 狀態重複，應該命中 |
| `pseudo_steering_beta` | 換 steering 狀態 beta，應該 miss |

## 測試環境

```text
Machine: Apple Silicon M1 Pro
Backend: llama.cpp local server
Model: Qwen3.5 4B GGUF, Q4_K_M style local quant
Context: 32768
Runtime mode: OpenAI-compatible /v1/chat/completions
Prompt length: about 6000 characters, 3808 prompt tokens
Output length: 64 tokens
Date: 2026-05-15
```

啟動 server 的方向：

```sh
python3 qwen_engine.py --port 18180
```

這個 repo 沒有放模型檔與 build 產物。模型與 llama.cpp build 請自行準備。

## 如何重跑

啟動本機 Qwen3.5 OpenAI-compatible server 後：

```sh
python3 scripts/qwen_prefix_cache_lab.py \
  --base-url http://127.0.0.1:18180/v1 \
  --model qwen3.5:4b \
  --prefix-chars 6000 \
  --max-tokens 64 \
  --trace-jsonl traces/prefix-cache-smoke.jsonl
```

如果你的 server 不支援 streaming usage，也可以先加：

```sh
--no-stream
```

## Smoke Result

本次在 M1 Pro 上的實測：

```text
cold_fixed_a TTFT:          10.524s, cached_tokens: 0
warm_fixed_a_1 TTFT:         0.101s, cached_tokens: 3804
warm_fixed_a_2 TTFT:         0.079s, cached_tokens: 3804
same_prefix_new_tail TTFT:   1.565s, cached_tokens: 3292
mutated_middle TTFT:        10.383s, cached_tokens: 0
pseudo_steering_alpha_1:    10.599s, cached_tokens: 0
pseudo_steering_alpha_2:     0.087s, cached_tokens: 3804
pseudo_steering_beta:       10.592s, cached_tokens: 0
```

完整 trace 放在：

```text
traces/prefix-cache-smoke-m1pro-2026-05-15.jsonl
```

## 判讀

### 1. 固定 prefix cache 有效

第一次跑：

```text
TTFT: 10.524s
cached_tokens: 0
```

第二次同樣 prompt：

```text
TTFT: 0.101s
cached_tokens: 3804
```

第三次：

```text
TTFT: 0.079s
cached_tokens: 3804
```

這代表固定 prefix 已經被重用。對本地 worker 來說，這是非常有用的訊號。

### 2. 同 prefix 換問題仍有部分命中

```text
same_prefix_new_tail TTFT: 1.565s
cached_tokens: 3292
```

這很符合實際 Agent / RAG 工作流：

```text
固定 SOP / 技能規則 / 品牌語氣
  + 每次不同任務
```

只要前面固定，後面換問題仍然能省掉大量 prefill。

### 3. prefix 中間變動會 miss

```text
mutated_middle TTFT: 10.383s
cached_tokens: 0
```

這是正確行為。  
如果 prefix 中間被改掉還命中，反而代表 cache key 或 checkpoint 邏輯有污染風險。

### 4. steering state 必須進 cache key

這次用 visible text 模擬：

```text
PSEUDO_STEERING_STATE=alpha
PSEUDO_STEERING_STATE=beta
```

結果：

```text
alpha 第一次: miss
alpha 第二次: hit
beta 第一次: miss
```

這對應 Antirez / DS4 討論的重點：  
如果未來真的有 vector steering、LoRA、policy vector、style vector，cache 不能只看 prompt 文字。

安全 cache key 至少要包含：

```text
model id
model build / commit
quant type
tokenizer hash
chat template hash
RoPE / context settings
prompt prefix hash
LoRA hash
steering vector hash
steering strength
policy hash
KV cache format version
runtime version
```

## Cache Metadata / Hash Lab

在真正做 SSD KV persistence 之前，必須先確認一件事：

> 任何會改變 KV cache 語義的狀態，都必須讓 cache key 改變。

所以 repo 另外加入了：

```text
scripts/qwen_cache_key_lab.py
```

這個測試不需要啟動模型 server，它只測 metadata/hash 設計是否安全。

重跑：

```sh
python3 scripts/qwen_cache_key_lab.py \
  --trace-json traces/cache-key-lab-2026-05-15.json
```

如果你要把本機 GGUF 也納入實際 hash：

```sh
python3 scripts/qwen_cache_key_lab.py \
  --model-path /path/to/model.gguf \
  --tokenizer-sha256 <tokenizer_sha256> \
  --runtime-version <llama.cpp_commit_or_build_id> \
  --trace-json traces/cache-key-lab-real-model.json
```

### Key Scope

這裡刻意分成兩種 key：

```text
prefix_key   = 用來重用固定 prefix 的 KV cache
request_key  = 用來辨識整次完整請求
```

這個差別很重要：

```text
同一個固定 SOP + 不同 user tail
```

應該是：

```text
prefix_key 不變
request_key 改變
```

也就是說，換問題不該毀掉可重用的固定 prefix cache；但完整請求本身當然要是不同 key。

### Metadata Hash Result

本次結果：

```text
identical_prefix_same_state: PASS
same_prefix_new_tail_prefix_key: PASS
same_prefix_new_tail_request_key: PASS
mutated_prefix_text: PASS
model_sha256: PASS
tokenizer_sha256: PASS
quant_type: PASS
runtime_version: PASS
chat_template: PASS
rope_settings: PASS
context_size: PASS
kv_cache_format_version: PASS
lora_sha256: PASS
policy_sha256: PASS
hidden_steering_state: PASS
steering_vector_sha256: PASS
steering_strength: PASS
```

完整 trace：

```text
traces/cache-key-lab-2026-05-15.json
```

這次測試也抓到一個設計細節：第一版 `prefix_key` 不小心把 full prompt hash 放進 key，導致「同 prefix 換 user tail」會錯誤 miss。修正後，prefix scope 不再納入 `prompt_full_sha256`，只有 request scope 會納入。

## 目前限制

這次實驗證明的是：

```text
Qwen3.5 + llama.cpp 可以吃到很強的 fixed-prefix prompt cache / checkpoint 效果。
```

還沒有證明的是：

```text
乾淨重啟 server 後，從 SSD 載入 KV cache 並繼續命中。
```

也就是說，目前還不是完整 DS4-style SSD KV persistence。

## 對小企業 Agent 的實際價值

最適合先用在：

- 社群內容再利用 SOP
- 日本熱門話題改寫 SOP
- n8n 自動化品質檢查
- Rooming list / OCR 後處理
- 固定工具 schema worker
- 固定品牌語氣與禁用規則
- RAG 熱門文件或 FAQ

實際架構可以是：

```text
固定 prefix:
  system prompt
  skill instructions
  brand voice
  QC rules
  tool schema
  common SOP

動態 tail:
  今日主題
  使用者輸入
  RAG top-k 摘要
  要輸出的格式
```

這樣每次只變 tail，prefix 就能吃 cache。

## 下一步

1. 測試 12k / 24k / 48k token prefix 的命中曲線。
2. 測試 server restart 後是否能用 slot save path 或外部 state 做恢復。
3. 把 metadata cache key 接到實際 KV 檔案命名與 header 驗證。
4. 比較 Qwen3.5 4B、Qwen3.6 27B MTP、Gemma 3 4B MTP、DS4 Flash。
5. 把這套接進實際 workflow，例如：
   - `japan-trend-fb-publisher`
   - `ai-trend-to-business-content`
   - rooming worker
   - n8n content repurpose worker

## 結論

這次測試證明：**本地小模型不是只能單次問答，它可以被設計成固定技能 worker。**

對個人和小企業來說，真正的落地方式不是追求最強 Agent，而是把常用 SOP 固定成 prefix，讓本地模型重複利用已經算過的上下文。

一句話：

> 固定的知識與流程，不要每次重算。  
> 把它變成本地 AI worker 的快取資產。
