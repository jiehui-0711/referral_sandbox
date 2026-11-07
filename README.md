# 醫療轉診沙盒

## 這個研究在做什麼

用一個醫療轉診沙盒評估 LLM Agent：任務進行到一半注入**擾動**（例如病人打電話要求改期），觀察 Agent 會不會修正那些由舊事實推導出來、已經**過期**的安排 —— 擾動事件本身就說出口的是**直接影響**，必須再查詢推導才會發現的是**間接影響**。

沙盒同時在世界各處植入**病人負擔**（往返次數、費用、交通時間），其中一項由病人自己提出、決定轉診走向，其餘藏在資訊段落裡，另外驗證 Agent 有沒有辨識並告知病人。

架構分四層（環境、互動、規範、任務），細節見 `沙盒架構層級.pdf`。

---

## 環境怎麼裝

只依賴 Python ≥ 3.10 與 `pyyaml`。推論走 HTTP 給 ollama，沙盒本身不載入任何模型。

```bash
conda env create -f environment.yml
conda activate rsandbox
```

`environment.yml` 裡已經含 `pip install -e .`。驗證：

```bash
python -m rsandbox load
```

要看到 `✓ 全部載入成功`。

如果不用 conda，或是遇到 `No module named 'yaml'`：

```bash
make deps        # 補 pyyaml 與 pytest
make install     # python -m pip install -e .
```

跑真模型另外需要 ollama：

```bash
ollama serve
ollama pull qwen2.5:7b
```

---

## 資料放哪

全部在 `src/rsandbox/data/`，純 YAML 與純文字，沒有程式碼。

| 路徑 | 內容 |
|---|---|
| `fixed/` | **固定層** —— 每個任務都會出現的部分 |
| `fixed/origin/C0.yaml` | 轉出端診所，Agent 所屬，不可轉介 |
| `fixed/institutions/H1–H4.yaml` | 四間轉入醫院 |
| `fixed/blocks.yaml` | 醫院層資訊段落 |
| `fixed/places.yaml` | 地點與單程交通時間 |
| `elements/` | **元素庫**（變動層）—— 13 個元素，各自帶科別、檢查項目、資訊段落、診次規則、指引 |
| `elements/COLO/guidelines/` | 醫療指引，`when … action …` 的條件式文件 |
| `cases/COLO-01/` | **情境** —— 一位病人的一次轉診需求 |
| `cases/COLO-01/case.yaml` | 情境定義、必要元素與干擾元素清單 |
| `cases/COLO-01/record.yaml` | 病歷 |
| `cases/COLO-01/patient_doc.yaml` | 病人的個人視角資料，餵給病人模型 |
| `cases/COLO-01/places.yaml` | 這個情境的地點與交通時間 |
| `cases/COLO-01/v*.yaml` | **六個變體**，各自的擾動事件與起始狀態 |
| `cases/COLO-01/scripts/v*.yaml` | 每個變體的**參考腳本**，以及幾條別的走法 |
| `tools/{global,origin,institution,department}/` | 工具定義，資料夾名是資訊層級的分組 |
| `prompts/referral_agent.txt` | 行政規則，所有任務共用同一份 |
| `prompts/patient_agent.txt` | 病人模型的行為規則 |
| `state/schema.yaml` | 狀態的欄位定義 |
| `truth/indication_class.yaml` | 評估用的對照資料 |

目前只有 `COLO-01` 一個情境，六個變體：

```
v0_clean           無擾動              腳本 38 步
v1_postponed       延後一週            腳本  9 步
v2_contact_only    只換電話            腳本  2 步
v3_earlier         要求提前            腳本  9 步
v4_slot_cancelled  醫院取消時段         腳本  9 步
v5_self_stopped    病人自行提前停藥      腳本  4 步
```

`v1`~`v5` 的起始狀態都是 `v0` 的參考腳本跑完之後的世界 —— 繼承的是**世界狀態**，不是對話歷史，所以模型看不到 v0 是怎麼做出來的。

執行結果寫到 `runs/`（已在 `.gitignore`）。

---

## 怎麼跑一個最小的例子

### 一、不接模型，用參考腳本走一遍

```bash
python -m rsandbox trace COLO-01/v1_postponed --patient-model none
```

這會把世界的反應逐步印出來，不判對錯。開頭長這樣：

```
  COLO-01/v1_postponed　腳本　reference
  9 次工具呼叫　1 個回合　29 列資料改動

─── 回合 1　2026-09-03　Thu ─────────────────────────
  事件│ patient_call
  觀察│ Today is 2026-09-03.
  觀察│ A referral case has been opened for patient P001.
  觀察│ The patient called and said: ... Could we move it to the 17th? ...

  呼叫│ query_slots(institution_id='H1', dept='GI', service='COL', ...)
  回覆│ ok　{"slots": [{"slot_id": "H1-GI-COL-20260917-AM", ...}]}
  呼叫│ reschedule_appointment(appointment_id='APT-001', ...)
  改動│ appointments.APT-001.date　'2026-09-10' → '2026-09-17'
```

結尾要看到 `每一次呼叫都 ok`。`--patient-model none` 表示不接病人模型，病人一律回固定的一句。

六個變體一起跑：

```bash
python -m rsandbox trace all --patient-model none
```

### 二、接真模型

```bash
python -m rsandbox check-llm --model qwen2.5:7b     # 確認接得上、支援 function calling

python -m rsandbox run COLO-01/v1_postponed \
    --model qwen2.5:7b --patient-model qwen2.5:7b \
    --max-steps 12 --num-ctx 16384
```

**`--num-ctx` 要給。** 系統提示詞加工具清單的固定開銷約 4,500 token，而 ollama 的預設 context window 是 4096 —— 不指定的話歷史會被服務端切掉，而且不回報、不報錯。執行結果那一行會印出送出量與服務端處理量，頂到上限會標 ⚠。

v0 的資料量比較大，用 `--num-ctx 32768`。

看一次執行的完整過程：

```bash
python -m rsandbox show-run <run_id>     # run_id 在 runs/ 底下
```

### 三、其他指令

```bash
python -m rsandbox show {tools,rules,slots,tasks,nodes}
python -m rsandbox show-prompt              # Agent 收到的系統提示詞
python -m rsandbox show-raw <run_id>        # 逐字印出原始訊息序列
python -m rsandbox call <task> <tool> ...   # 手動打一個工具
```

---

## 檢查

```bash
python -m pytest checks/tests/ -q     # 程式行為測試
python checks/agent_visible.py        # 掃一遍 Agent 讀得到的所有字串
```

`checks/` 整個刪掉，沙盒照常運作 —— 設計上檢查不住在架構裡。

---

## 還沒做的

- **評估尚未實作**。目前只有 `trace`（印過程）與 `run`（跑真模型），沒有「最終資料庫與預期資料庫逐欄比對」的指令，病人負擔的斷言檢查也還沒寫。
- 只有 `COLO-01` 一個情境。
- 節點圖還在資料裡，但 `move_to` 已關閉、工具的位置權限已放寬，所以位置目前不影響任何事。
