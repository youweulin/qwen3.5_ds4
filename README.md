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

## KV Artifact Naming / Header Lab

metadata key 通過後，下一步是把它接到實際 KV 檔案外殼：

```text
cache key -> KV artifact filename
cache metadata -> KV artifact header
KV bytes -> payload with sha256 digest
```

repo 加入了：

```text
scripts/qwen_kv_artifact_lab.py
```

這個測試還不寫入真的 llama.cpp KV tensor，而是使用 synthetic KV bytes。  
它的目的不是測速度，而是測「未來真的把 KV bytes 放進來時，檔案能不能安全驗證」。

Artifact 格式：

```text
MAGIC:        QWEN35DS4KV1\0
HEADER_LEN:   8-byte big-endian unsigned integer
HEADER_JSON:  canonical JSON
PAYLOAD:      KV bytes, currently synthetic bytes
```

檔名：

```text
{cache_scope}-{cache_key}.qkv
```

header 會包含：

```text
artifact_schema
key_domain
cache_key
cache_scope
metadata
payload_sha256
payload_size
created_unix
created_by
```

驗證時會檢查：

```text
1. 檔名 key == header cache_key
2. header metadata 重新計算出的 key == header cache_key
3. payload sha256 == header payload_sha256
4. payload size == header payload_size
5. 如果有 expected metadata，expected key 必須一致
```

重跑：

```sh
python3 scripts/qwen_kv_artifact_lab.py \
  --clean \
  --trace-json traces/kv-artifact-lab-2026-05-15.json
```

### KV Artifact Result

本次結果：

```text
valid_artifact: PASS
filename_key_mismatch: PASS
payload_tamper: PASS
header_metadata_tamper: PASS
expected_metadata_mismatch: PASS
```

錯誤驗證也符合預期：

```text
filename_key_mismatch -> filename_key_matches_header
payload_tamper -> payload_sha256_matches
header_metadata_tamper -> header_key_matches_metadata, expected_metadata_key_matches
expected_metadata_mismatch -> expected_metadata_key_matches
```

完整 trace：

```text
traces/kv-artifact-lab-2026-05-15.json
```

這一步的意義：

```text
現在已經不是只有「算出 cache key」。
而是已經有一個能放在 SSD 上的 KV 檔案外殼：
檔名可索引、header 可驗證、payload 可防竄改、metadata 可拒用錯狀態。
```

下一步是把 synthetic payload 換成真正 runtime 存下來的 KV bytes，並驗證抽出後可以 restore。

## llama.cpp Slot Bytes Artifact Lab

這一步已經把 synthetic payload 換成 llama.cpp 實際存出的 slot/state bytes。

新增腳本：

```text
scripts/qwen_llamacpp_slot_artifact_lab.py
```

測試流程：

```text
1. 用 /completion 對 slot 0 跑一段 6000 字固定 prefix
2. 呼叫 POST /slots/0?action=save
3. 讀取 llama.cpp 實際寫出的 raw slot 檔
4. 把 raw bytes 包進 .qkv artifact header
5. 驗證 artifact filename/header/metadata/payload sha256
6. 從 .qkv 抽出 payload，寫回 raw slot 檔
7. 呼叫 POST /slots/0?action=restore
8. 同 prefix 換問題，確認 restore 後仍能吃 cache
```

啟動 llama-server 時要注意 `--slot-save-path`：

```text
llama.cpp server 目前是用 slot_save_path + filename 直接相加。
所以 slot save path 建議用尾端帶 / 的路徑。
```

重跑測試：

```sh
python3 scripts/qwen_llamacpp_slot_artifact_lab.py \
  --base-url http://127.0.0.1:18180 \
  --slot-save-path "$PWD/artifacts/llamacpp-slots/" \
  --artifact-dir artifacts/llamacpp-slot-artifacts \
  --trace-json traces/llamacpp-slot-artifact-lab-2026-05-15.json
```

### llama.cpp Slot Artifact Result

本次 M1 Pro + Qwen3.5 4B 實測：

```text
raw_slot_size:      178,449,080 bytes
raw_slot_mib:       170.18 MiB
raw_slot_sha256:    6928ec5a2627a91aba77d4f91fbf671b1e1a4d2a682c969b600e3bbc78c1a984
slot_save:          n_saved 3835, n_written 178,449,080, save_ms 198.880
slot_restore:       n_restored 3835, n_read 178,449,080, restore_ms 23.436
initial_prompt_n:   3820
followup_prompt_n:  518
```

檢查結果：

```text
raw_payload_matches_artifact_payload: PASS
artifact_payload_matches_extracted_slot: PASS
artifact_verify_ok: PASS
restore_reported_tokens: PASS
restored_cache_effective: PASS
```

完整 trace：

```text
traces/llamacpp-slot-artifact-lab-2026-05-15.json
```

判讀：

```text
.qkv artifact 裡已經不是假 bytes，而是 llama.cpp 實際 save 出來的 slot bytes。
從 .qkv 抽出 payload 後，restore 回 llama.cpp slot 成功。
restore 後同 prefix 換問題，prompt_n 從 3820 降到 518，代表 cache 狀態真的被帶回來。
```

目前限制：

```text
這還是 llama.cpp 的 whole slot state，不是乾淨切好的 prefix-only KV slice。
但它已經證明 metadata/header/hash 外殼可以承載真實 runtime bytes，
而且 payload 抽出後能被 llama.cpp restore 使用。
```

## External Cache Manager Proxy Lab

在改 llama.cpp C++ 之前，先測「A 路線」：

```text
Client / Agent
  -> Python cache manager
  -> llama.cpp /completion + /slots save/restore
```

新增腳本：

```text
scripts/qwen_cache_manager_proxy_lab.py
```

這個腳本不是正式 HTTP proxy，而是把 proxy policy 做成可量測的 lab：

```text
1. baseline：不用 cache manager，長 prefix 直接冷跑
2. manager miss：prefix key 不存在，正常 completion，跑完 save slot 並包成 .qkv
3. erase active slot，模擬切換任務或 session
4. manager hit：prefix key 已存在，從 .qkv 抽出 payload，restore slot，再跑 follow-up
```

重跑：

```sh
python3 scripts/qwen_cache_manager_proxy_lab.py \
  --base-url http://127.0.0.1:18180 \
  --slot-save-path "$PWD/artifacts/proxy-cache-slots/" \
  --artifact-dir artifacts/proxy-cache-artifacts \
  --trace-json traces/cache-manager-proxy-lab-2026-05-15.json \
  --clean
```

### Proxy Lab Result

本次 M1 Pro + Qwen3.5 4B 實測：

```text
baseline no-manager cold follow-up:
  latency:   11.121s
  prompt_n: 3847

manager miss then save:
  completion latency: 11.031s
  save latency:        0.172s
  total latency:      11.203s
  prompt_n:           3845
  slot payload:        179,268,880 bytes

manager hit then restore:
  restore latency:     0.023s
  completion latency:  2.072s
  total latency:       2.095s
  prompt_n:            518

speedup vs cold:       5.31x
prompt tokens saved:   3329
```

完整 trace：

```text
traces/cache-manager-proxy-lab-2026-05-15.json
```

判讀：

```text
A 路線的外部 manager 在長 prefix 場景是值得的。
Python 代理層與 metadata/hash/header 驗證不是主要瓶頸。
真正成本是第一次 prefill 與 save；命中後 restore 只有約 23ms。
```

也就是說：

```text
短 prompt 不應該進 SSD cache manager。
長 SOP / skill / glossary / 100k context 才值得。
```

## Workflow Policy Bench

接著把同一套 external cache manager policy 套到三種本地 agent workflow：

```text
1. FB content SOP
   固定品牌語氣 + 發文規則 + 禁用連結規則

2. Translation glossary
   固定術語表 + 台灣語氣 + 品質規則

3. Rooming list QC
   固定航空/房型/姓名檢查規則
```

新增腳本：

```text
scripts/qwen_workflow_policy_bench.py
```

這個測試的目的不是評估 Qwen3.5 4B 的最終內容品質，而是先用 4B 測：

```text
cache policy 是否通用
不同 workflow 的 hit/miss 是否穩定
save/restore 成本
prompt_n 可以省多少
```

同一套 policy：

```text
cache_scope: prefix
prefix_chars: 6000
n_predict: 16
miss: 先正常 completion，跑完 save slot，包成 .qkv
hit: 從 .qkv 抽 payload，restore slot，再跑 follow-up task
```

重跑：

```sh
python3 scripts/qwen_workflow_policy_bench.py \
  --base-url http://127.0.0.1:18180 \
  --slot-save-path "$PWD/artifacts/workflow-policy-slots/" \
  --artifact-dir artifacts/workflow-policy-artifacts \
  --trace-json traces/workflow-policy-bench-2026-05-15.json \
  --clean
```

### Workflow Policy Result

本次 M1 Pro + Qwen3.5 4B 實測：

```text
FB content SOP:
  baseline cold:       9.085s
  manager hit total:   2.025s
  restore:             0.019s
  prompt_n:            3142 -> 507
  prompt_n_saved:      2635
  speedup:             4.49x

Translation glossary:
  baseline cold:       9.068s
  manager hit total:   2.069s
  restore:             0.021s
  prompt_n:            3146 -> 513
  prompt_n_saved:      2633
  speedup:             4.38x

Rooming list QC:
  baseline cold:       9.151s
  manager hit total:   2.014s
  restore:             0.020s
  prompt_n:            3193 -> 516
  prompt_n_saved:      2677
  speedup:             4.54x

Average speedup:       4.47x
```

完整 trace：

```text
traces/workflow-policy-bench-2026-05-15.json
```

判讀：

```text
同一套 prefix cache policy 對三種固定 SOP workflow 都有效。
4B 不適合用來判斷文案、翻譯、rooming QC 的最終品質，
但很適合用來測速度、cache hit、restore 成本與 prompt_n 節省。
```

下一步：

```text
用同一套 workflow policy 換更強模型測品質，例如 Qwen3.6 27B / DS4 Flash / Gemma 3 31B。
速度/cache 用 4B 建 baseline；品質用大模型判斷是否能產品化。
```

## Memory Policy Bench

為了精準回答「到底省多少 RAM」，新增：

```text
scripts/qwen_memory_policy_bench.py
```

測試流程：

```text
1. 腳本啟動 llama-server
2. 記錄 baseline RSS / llama.cpp Metal memory breakdown
3. 跑 workflow A
4. save workflow A slot 成 .qkv artifact
5. erase active slot
6. 跑 workflow B，模擬切換工作流
7. restore workflow A
8. 跑 workflow A follow-up，確認 cache hit
```

重跑：

```sh
python3 scripts/qwen_memory_policy_bench.py \
  --slot-save-path "$PWD/artifacts/memory-policy-slots/" \
  --artifact-dir artifacts/memory-policy-artifacts \
  --trace-json traces/memory-policy-bench-2026-05-15.json
```

### Memory Result

本次 M1 Pro + Qwen3.5 4B + `ctx-size 32768` 實測。

llama.cpp 啟動時的靜態 memory breakdown：

```text
metal_model_mib:       2584.74 MiB
metal_kv_mib:          1024.00 MiB
metal_recurrent_mib:     50.25 MiB
metal_compute_mib:      490.00 MiB
metal_static_total:    4148.99 MiB

cpu_model_mib:          497.31 MiB
cpu_compute_mib:         74.02 MiB
```

RSS samples:

```text
server_ready_baseline:          3912.86 MiB
after_workflow_a_completion:    3842.47 MiB
after_workflow_a_save_artifact: 3739.38 MiB
after_erase_workflow_a_slot:    3739.78 MiB
after_workflow_b_completion:    3840.89 MiB
after_restore_workflow_a_slot:  3842.94 MiB
after_workflow_a_followup_hit:  3846.50 MiB
```

Workflow A artifact:

```text
artifact_size: 149.23 MiB
save:          3165 tokens, 156,478,440 bytes, 266.726ms
restore:       3165 tokens, 156,478,440 bytes, 19.372ms
follow-up:     prompt_n 503, latency 1.991s
```

完整 trace：

```text
traces/memory-policy-bench-2026-05-15.json
```

判讀：

```text
erase/save 不會讓 RSS 大幅歸零，因為 llama.cpp 會預配置 active ctx/KV buffer。
這次 32768 ctx 的 Metal KV buffer 本身就是 1024 MiB。
```

所以這套目前省的不是：

```text
單一 active workflow 的 llama.cpp KV buffer
```

而是：

```text
不用讓多個 workflow / skill 的已算狀態同時 active 常駐。
不用切回 workflow A 時重新 prefill。
```

換句話說：

```text
active RAM/VRAM 仍要保留目前正在工作的 context。
冷掉的 workflow state 可以放 SSD artifact。
切回來用 19ms restore，避免 9s prefill。
```

## DS4-style Runtime Cache Manager Lab

在參考 DS4 之後，新增第一版 B 路線原型：

```text
scripts/qwen_ds4_runtime_cache_manager_lab.py
```

這一步先不改 llama.cpp C++，而是在 runtime manager 層把 DS4 的政策做出來：

```text
runtime 內建自動 cache manager
prompt prefix hash
metadata header
save/restore llama.cpp slot
cold / continued / evict policy
disk budget eviction
```

它和 DS4 一樣是「一個 live session + SSD checkpoint」思路，不是把 active KV buffer 直接搬到 SSD 分頁執行。  
所以目前省的是切換 workflow 時的重算時間，不是單一 active context 的 KV RAM。

重跑 dry-run policy 測試：

```sh
python3 scripts/qwen_ds4_runtime_cache_manager_lab.py \
  --dry-run \
  --clean \
  --disk-budget-mib 5 \
  --dry-run-payload-bytes 2097152 \
  --trace-json traces/runtime-cache-manager-lab-dryrun-2026-05-15.json
```

接真實 llama-server 時：

```sh
python3 scripts/qwen_ds4_runtime_cache_manager_lab.py \
  --slot-save-path "$PWD/artifacts/runtime-cache-slots/" \
  --artifact-dir artifacts/runtime-cache-artifacts \
  --trace-json traces/runtime-cache-manager-lab-2026-05-15.json \
  --sequence fb,translation,fb,rooming,translation,fb
```

### Runtime Manager Dry-run Result

本次 dry-run 故意把 disk budget 設小，確認 eviction policy 會啟動：

```text
requests:              6
cache_hits:            2
cache_misses:          4
saves_by_reason:       cold 4, evict 1, continued 1
disk_budget_evictions: 2
artifact_count:        2
artifact_mib:          4.00
```

Request flow：

```text
fb:          cold -> cold_save
translation: cold -> cold_save
fb:          disk_hit_restore
rooming:     evict current fb -> cold -> cold_save -> disk_budget_evict
translation: cold again because previous translation artifact was evicted
fb:          disk_hit_restore -> continued_save
```

完整 trace：

```text
traces/runtime-cache-manager-lab-dryrun-2026-05-15.json
```

這代表現在已經有 DS4-style 的政策骨架：

```text
miss 時 cold save
切換 workflow 前 evict save
回到舊 workflow 時 disk restore
命中到一定次數後 continued save
超過磁碟預算後淘汰低價值 artifact
```

### Runtime Manager Real llama.cpp Result

真實啟動 llama-server 後，跑同一套 sequence：

```text
fb,translation,fb,rooming,translation,fb
```

本次 M1 Pro + Qwen3.5 4B + `ctx-size 32768` 實測：

```text
requests:              6
cache_hits:            3
cache_misses:          3
saves_by_reason:       cold 3, evict 2, continued 1
disk_budget_evictions: 0
artifact_count:        3
artifact_mib:          448.97
```

每個 request：

```text
fb cold:
  prompt_n:    3150
  completion:  9.117s

translation cold:
  prompt_n:    3148
  completion:  9.035s

fb disk restore:
  restore:     0.020s
  prompt_n:    509
  completion:  2.018s

rooming cold:
  prompt_n:    3193
  completion:  9.200s

translation disk restore:
  restore:     0.023s
  prompt_n:    468
  completion:  1.937s

fb disk restore:
  restore:     0.022s
  prompt_n:    464
  completion:  1.923s
```

完整 trace：

```text
traces/runtime-cache-manager-lab-real-2026-05-15.json
```

這次也抓到一個重要限制：llama.cpp whole-slot checkpoint 不是 DS4 那種乾淨 prefix payload。  
如果把 `evict` 或 `continued` checkpoint 覆蓋成正式 lookup artifact，可能會把已生成的尾巴一起存進去，讓下一次 prefix hit 變成假命中。

所以目前 manager 的安全做法是：

```text
cold save -> 正式 prefix lookup artifact
evict / continued / shutdown save -> session-checkpoints/，預設不覆蓋 prefix lookup artifact
```

這讓 runtime 仍保留 DS4-style event policy，但避免污染 prefix cache。

### 和 DS4 的差異

DS4 的 payload 是它自己的 DeepSeek V4 Flash session state，包含壓縮 KV rows、frontier、token IDs、logits 等。  
我們目前的 payload 是 llama.cpp `/slots?action=save` 存出的 whole slot state。

所以目前還有兩個限制：

```text
1. artifact 是 whole slot，不是乾淨的 prefix-only KV slice。
2. restore 後仍依賴 llama.cpp cache_prompt 去對齊共同 prefix。
```

但這已經足夠先驗證產品政策：

```text
哪些 workflow 值得 cache
disk budget 應該怎麼設
切換 skill/agent 時能省多少 prefill
哪些 metadata 必須進 key 才安全
```

### KV Artifact Performance

為了確認這層外殼本身不會變成瓶頸，另外加入：

```text
scripts/qwen_kv_artifact_perf_lab.py
```

測試項目：

```text
1. synthetic KV payload generation
2. header/key/digest creation
3. file write
4. file read + header verification + payload sha256
```

重跑：

```sh
python3 scripts/qwen_kv_artifact_perf_lab.py \
  --clean \
  --sizes 4kb,1mb,16mb \
  --rounds 3 \
  --trace-json traces/kv-artifact-perf-2026-05-15.json
```

本次結果：

```text
4KB:
  write_avg_s: 0.000s
  verify_avg_s: 0.000s

1MB:
  write_avg_s: 0.001s
  verify_avg_s: 0.001s
  write_mib_s: 1039.343
  verify_mib_s: 1059.278

16MB:
  write_avg_s: 0.014s
  verify_avg_s: 0.011s
  write_mib_s: 1250.678
  verify_mib_s: 1448.827
```

再補一個 64MB 單輪：

```text
64MB:
  write_avg_s: 0.075s
  verify_avg_s: 0.043s
  write_mib_s: 855.696
  verify_mib_s: 1491.728
```

Trace files:

```text
traces/kv-artifact-perf-2026-05-15.json
traces/kv-artifact-perf-64mb-2026-05-15.json
```

判讀：

```text
這階段實裝的安全外殼成本很低。
對 16MB / 64MB 等級 payload，write + verify 都是毫秒到數十毫秒級。
未來真正瓶頸更可能是 runtime KV serialization / deserialization，而不是 header/hash/filename 驗證。
```

## Translation Workload Benchmark

第一階段 smoke test 證明固定 prefix 會命中，但真正落地時更重要的是：

> 大量翻譯文章時，固定翻譯 SOP / 術語表 / 品質規則能不能省時間？

所以 repo 加入第三個測試：

```text
scripts/qwen_translation_cache_bench.py
```

這個 benchmark 使用合成日文文章段落，避免版權問題，並比較三種做法：

| Group | 說明 |
| --- | --- |
| `cached_long_sop` | 固定 6000 字翻譯 SOP + 術語表，可吃 prefix cache |
| `busted_long_sop` | 每次在 SOP 開頭加入不同 marker，模擬動態 prompt 污染，吃不到 cache |
| `minimal_sop` | 很短的翻譯 prompt，速度快但缺少術語表與品質控管 |

重跑：

```sh
python3 scripts/qwen_translation_cache_bench.py \
  --base-url http://127.0.0.1:18180/v1 \
  --model qwen3.5:4b \
  --segments 4 \
  --sop-chars 6000 \
  --max-tokens 128 \
  --trace-jsonl traces/translation-cache-bench-m1pro-2026-05-15.jsonl
```

### Translation Result

本次 M1 Pro 實測：

```text
cached_long_sop[0]: TTFT 5.350s, latency 9.751s, cached_tokens 0
cached_long_sop[1]: TTFT 1.465s, latency 5.889s, cached_tokens 1436
cached_long_sop[2]: TTFT 1.462s, latency 5.756s, cached_tokens 1436
cached_long_sop[3]: TTFT 1.480s, latency 5.868s, cached_tokens 1436

busted_long_sop[0]: TTFT 5.452s, latency 9.892s, cached_tokens 0
busted_long_sop[1]: TTFT 5.567s, latency 10.035s, cached_tokens 0
busted_long_sop[2]: TTFT 5.485s, latency 9.867s, cached_tokens 0
busted_long_sop[3]: TTFT 5.548s, latency 9.993s, cached_tokens 0

minimal_sop avg: TTFT 0.421s, latency 4.775s, cached_tokens 0
```

Summary:

```text
cached_long_sop warm TTFT avg: 1.469s
busted_long_sop warm TTFT avg: 5.533s
TTFT speedup: 3.77x

cached_long_sop warm latency avg: 5.838s
busted_long_sop warm latency avg: 9.965s
end-to-end speedup: 1.71x
```

Trace files:

```text
traces/translation-cache-bench-m1pro-2026-05-15.jsonl
traces/translation-cache-bench-m1pro-2026-05-15.summary.json
```

### Translation Benchmark Takeaway

`minimal_sop` 最快，因為它幾乎沒有前置規則要讀；但它不是同等品質對照。  
它缺少：

- 固定術語表
- 台灣用語規則
- 品牌語氣
- 不杜撰規則
- 社群可讀性規則
- 品質管理要求

真正有產品價值的是 `cached_long_sop`：

```text
第一次讀完整 SOP 比較慢
後續每段文章沿用同一套翻譯規則
TTFT 從 5.5s 左右降到 1.5s 左右
輸出品質規則仍然保留
```

這很適合內容工廠型任務：

```text
固定翻譯 SOP / glossary / style guide
  + 大量文章段落
  + 人工審稿
```

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
2. 用更長文章測試翻譯吞吐量，例如 20 / 50 / 100 段。
3. 測試 server restart 後是否能用 slot save path 或外部 state 做恢復。
4. 把 metadata cache key 接到實際 KV 檔案命名與 header 驗證。
5. 比較 Qwen3.5 4B、Qwen3.6 27B MTP、Gemma 3 4B MTP、DS4 Flash。
6. 把這套接進實際 workflow，例如：
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
