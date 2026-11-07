"""指令介面
  rsandbox load          把全部資料載入，印出載到了多少東西
  rsandbox show          看載入結果：工具、指引、診次、任務、節點
  rsandbox call          手動打一個工具，看信封長什麼樣
  rsandbox trace         跑一輪，把過程一步一步印出來（不做任何判定）
  rsandbox run           用真模型跑一個案例，把逐字稿寫成檔案
  rsandbox check-llm     確認模型接得上、支援原生 function calling
  rsandbox show-run      把一次已存檔的執行整理成人看得懂的樣子
  rsandbox show-raw      逐字印出原始訊息序列
  rsandbox show-prompt   印出 Agent 收到的系統提示詞

這裡只負責呼叫、排版、回報，真正的邏輯在 loaders / runtime 裡。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .loaders import load_all
from .runtime import Session
from .runtime import clock


#  load

def cmd_load(a) -> int: #載入全部資料，印出規模
    """cmd_load（把全部資料載入，印出載到了多少東西）。
    載入途中任何一個檔有問題就印出原因並回傳 1。
    """
    try:
        d = load_all()
    except Exception as e:                      # noqa: BLE001
        print(f"✗ 載入失敗：{e}")
        return 1

    print(f"  元素庫 {len(d['elements'])} 個元素"
          f"　任務組 {len(d['worlds'])} 組")
    for cid, w in sorted(d["worlds"].items()):
        print(f"    {cid}: 機構 {len(w['institutions'])}"
              f"　科別 {len(w['departments'])}"
              f"　項目 {len(w['services'])}"
              f"　段落 {len(w['blocks'])}"
              f"　診次 {len(w['slots'])}"
              f"　元素 {len(w['_elements'])}")
    print(f"  指引 {len(d['guidelines'])} 份")
    print(f"  工具 {len(d['tools'])}　變體 {len(d['tasks'])}")
    print("\n✓ 全部載入成功")
    return 0



#  演病人的那個模型

# PATIENT_MODEL_DEFAULT：病人預設用哪個模型。
# 跟受測模型分開設定：換受測模型的時候病人的行為不該跟著變。
PATIENT_MODEL_DEFAULT = "qwen2.5:7b"


def _patient_backend(a):
    """_patient_backend（照命令列參數做出演病人的後端）。
      - --patient-model none 就不接模型，病人一律回一句固定的
      - num_ctx 跟被測 Agent 用同一個值：同一個模型用兩個不同的
        context 大小，服務端會在兩邊來回的時候反覆卸載重載
    """
    from .agents.backends import build_backend
    model = getattr(a, "patient_model", None) or PATIENT_MODEL_DEFAULT
    if model == "none":
        return None
    return build_backend(getattr(a, "patient_provider", None) or "ollama",
                         model,
                         getattr(a, "patient_base_url", None),
                         getattr(a, "patient_api_key", None),
                         seed=getattr(a, "seed", 0), temperature=0.0,
                         num_ctx=getattr(a, "num_ctx", None) or None)


def _add_patient_args(sub) -> None:
    """_add_patient_args（把病人模型的參數掛到一個子指令上）。"""
    sub.add_argument("--patient-model", default=None,
                     help=f"演病人的模型（預設 {PATIENT_MODEL_DEFAULT}，"
                          f"none = 不接模型）")
    sub.add_argument("--patient-provider", default=None,
                     choices=["ollama", "openai"])
    sub.add_argument("--patient-base-url", default=None)
    sub.add_argument("--patient-api-key", default=None)


# NUM_CTX_DEFAULT（送給後端的 context window 上限預設值）。
NUM_CTX_DEFAULT = 16384


def _add_ctx_arg(sub) -> None:
    """_add_ctx_arg（把 context 上限的參數掛到一個子指令上）。
      - 預設 NUM_CTX_DEFAULT
      - 給 0 就不送這個參數，由服務端自己決定上限
    """
    sub.add_argument("--num-ctx", type=int, default=NUM_CTX_DEFAULT,
                     help=f"context window 上限。預設 {NUM_CTX_DEFAULT}，"
                          f"0 = 用服務端預設")


def _add_tool_call_arg(sub) -> None:
    """_add_tool_call_arg（把呼叫模式的參數掛到一個子指令上）。
      - 可選值與各自的意思寫在 agents/react.py 的 TOOL_CALL_MODES
      - 預設 one
    """
    from .agents.react import TOOL_CALL_MODES
    sub.add_argument("--tool-calls", default="one", choices=list(TOOL_CALL_MODES),
                     help="一則訊息給好幾個呼叫時：one 只跑第一個，"
                          "batch 依序跑完整批。預設 one")

#  show

def cmd_show(a) -> int: #查看工具、規則、診次、任務、節點——程式載入之後的形式
    """cmd_show（把載入後的資料印出來看）。
    它看到的是程式眼中的資料，不是 YAML 原文。
    所以推導出來的東西（展開後的診次、自動產生的節點樹、
    從指引複製過來的合法值）在這裡才看得到。
    """
    d = load_all()
    what = a.what
    if what == "tools":
        for tid, t in sorted(d["tools"].items()):
            params = ", ".join(f"{n}:{s.get('type')}" for n, s in (t.get("params") or {}).items())
            # 印可用節點的種類而不是逐一列出節點。訂號那支工具在
            # 每一個科別節點都能用，逐一列會爆版。
            print(f"{t['kind']:8} {tid:24} {t.get('gate','?'):11} ({params})")
    elif what == "rules":
        # 依它管哪一個決定分群印。同一個決定底下的幾份，
        # Agent 必須靠條件才分得出該用哪一份。
        from .loaders.guidelines import by_decision
        for decision, docs in sorted(by_decision(d["guidelines"]).items()):
            print(f"\n[{decision}]　{len(docs)} 份")
            for g in docs:
                print(f"  {g['guideline_id']:9} "
                      f"{json.dumps(g['when'], ensure_ascii=False)}")
                print(f"  {'':9} ->  "
                      f"{json.dumps(g['action'], ensure_ascii=False)}")
    elif what == "slots":
        # 診次跟著世界走，所以要說是哪一組的。
        for cid, w in sorted(d["worlds"].items()):
            print(f"\n── {cid}（{len(w['slots'])} 筆）──")
            for sid, row in sorted(w["slots"].items()):
                print(f"{sid:30} {row['dept_id']:9} {row['slot_type']:10}"
                      f" {row['date']}  {row['booked_count']}/{row['capacity']}"
                      f"  {row['status']}")
    elif what == "tasks":
        # 依組分群印。九十組的時候這是唯一讀得下去的排版。
        from .loaders.cases import cases_of
        for cid, variants in cases_of(d["tasks"]).items():
            head = d["tasks"][variants[0]]
            print(f"\n{cid}　{head['case_title']}"
                  f"　病人 {head['patient_id']}　{len(variants)} 個變體")
            for tid in variants:
                t = d["tasks"][tid]
                ev = len(t.get("event_script") or [])
                sc = d["scripts"].get(tid) or {}
                # 有沒有腳本要印出來，腳本是選填的，沒有也是合法的任務
                mark = (f"腳本 {len(sc.get('reference') or [])} 步"
                        f"＋{len(sc.get('alternatives') or [])} 條別的走法"
                        if sc.get("reference") else "沒有腳本")
                print(f"  {t['variant']:18} {t['title']:12}"
                      f" {t['clock']['start_date']} → {t['clock']['end_date']}"
                      f"  事件 {ev}　{mark}")
        print()
    elif what == "nodes":
        # 節點樹跟著世界走，所以要說是哪一組的。這裡印第一組的。
        cid = sorted(d["node_trees"])[0]
        print(f"（{cid} 這一組的節點樹。不同組選的元素不同，節點也不同）\n")
        for nid, n in sorted(d["node_trees"][cid].items()):
            print(f"{nid:16} {n['type']:11} tools={len(n['tools']):2}"
                  f"  children={len(n['edges'])}")
    return 0


#  call

def cmd_call(a) -> int: #手動打一個工具。除錯用，也用來手動確認不變量。
    """cmd_call（手動打一支工具，看信封長什麼樣）。
    除錯用，也用來手動確認不變量
    （例如傳一個亂七八糟的停藥日期，確認它真的照樣成功）。

    範例：
        rsandbox call COLO-01/v0_clean get_patient_record '{"patient_id": "P001"}'
        rsandbox call COLO-01/v0_clean move_to '{"node": "H1"}'

    每次呼叫都是全新的 Session，所以沒辦法先移動、再查那裡的東西，
    第二次呼叫又回到起點了。要看連續動作請用 trace。
    """
    d = load_all()
    s = Session(d, d["tasks"][a.task])
    args = json.loads(a.args) if a.args else {}
    env = s.call(a.tool, args)
    print(json.dumps(env, ensure_ascii=False, indent=2, default=str))
    return 0 if env["ok"] else 1


#  trace

def _inherited(d, task) -> tuple[int, int]:
    """_inherited（這個變體的起始狀態裡已經有幾筆紀錄），回 (呼叫, 改動)。
      - 起始狀態不是空的變體，兩份紀錄一開始就有列數
      - 報表從這兩個位置之後才算
    """
    from .runtime import initial
    st = initial.build(d, task).state
    return len(st["tool_call_log"]), len(st["change_log"])


def cmd_trace(a) -> int: #跑一輪並逐步印出過程，不做任何判定
    """cmd_trace（跑一輪，把過程一步一步印出來）。
      rsandbox trace COLO-01/v1_postponed            一個變體
      rsandbox trace COLO-01                         一整組的全部變體
      rsandbox trace all                             全部組的全部變體
      rsandbox trace COLO-01/v1_postponed -t <走法編號>  某一條別的走法
      rsandbox trace COLO-01/v1_postponed --model qwen3:8b   換真模型跑
      rsandbox trace COLO-01/v0_clean --full         工具回傳不截斷

      - 它不判對錯，沒有分數、沒有失敗種類，只回答這一輪實際發生了
        什麼：哪一天、模型說了什麼、呼叫哪支工具、環境回什麼、
        哪些資料真的被改動
      - 改動那幾行是直接讀變更紀錄（change_log），不是從工具的回傳值
        猜的
    """
    from .loaders.cases import cases_of

    d = load_all()
    groups = cases_of(d["tasks"])
    # 一個參數三種意思：
    #   all                     全部組的全部變體
    #   COLO-01                 這一組的全部變體
    #   COLO-01/v1_postponed    單一變體
    if a.task == "all":
        targets = sorted(d["tasks"])
    elif a.task in groups:
        targets = groups[a.task]
    elif a.task in d["tasks"]:
        targets = [a.task]
    else:
        print(f"✗ 沒有叫 {a.task!r} 的任務或任務組。"
              f"用 rsandbox show tasks 看有哪些")
        return 1
    if len(targets) > 1 and a.trajectory:
        print("✗ -t 只能跟單一變體一起用（別的走法是某個變體專屬的）")
        return 1
    for tid in targets:
        rc = _trace_one(d, tid, a)
        if rc:
            return rc
    return 0


def _trace_one(d, tid, a) -> int: #跑一個變體並印出來
    """_trace_one（跑一個變體，把過程印出來）。"""
    from .runtime import loop

    task = d["tasks"][tid]

    if a.model:
        from .agents.backends import build_backend
        from .agents.react import ReActAgent
        backend = build_backend(a.provider, a.model, a.base_url, a.api_key,
                                seed=a.seed, temperature=a.temperature,
                                num_ctx=a.num_ctx or None)
        agent = ReActAgent(backend, max_steps=a.max_steps,
                           tool_calls=a.tool_calls)
        who = f"真模型　{a.provider}:{a.model}"
    else:
        from .agents.scripted import ScriptedAgent
        sc = d["scripts"].get(tid) or {}
        if a.trajectory:
            alts = sc.get("alternatives") or []
            steps = next((x["steps"] for x in alts
                          if x["id"] == a.trajectory), None)
            if steps is None:
                names = [x["id"] for x in alts]
                print(f"✗ {tid} 沒有叫 {a.trajectory!r} 的走法"
                      + (f"，有的是：{'、'.join(names)}" if names else ""))
                return 1
        else:
            steps = sc.get("reference")
            if not steps:
                # 沒有腳本不是錯誤，只是沒東西可跑。
                print(f"\n（{tid} 沒有腳本，跳過。用 --model 可以讓真模型跑）")
                return 0
        agent = ScriptedAgent(steps)
        who = f"腳本　{a.trajectory or 'reference'}"

    inherited, _ = _inherited(d, task)
    s = loop.run_task(d, task, agent,
                      patient_backend=_patient_backend(a))
    _print_trace(s, tid, who, agent, full=a.full, inherited=inherited)
    return 0


def _print_trace(s, tid, who, agent, full=False, inherited=0) -> None: #把跑完的 Session 印成逐步紀錄
    """_print_trace（把一個跑完的 Session 印成逐步紀錄）。
    真模型跑的時候用它自己的逐字稿（裡面有它說的話）；
    腳本跑的時候逐字稿是空的，就從觀察與呼叫紀錄合成同一種版面。
    """
    cut_at = 100_000 if full else 300
    # 只印這一輪的呼叫。繼承來的那些屬於上一題。
    calls = s.state["tool_call_log"][inherited:]

    # 依呼叫序號把變更紀錄分組：這一次呼叫到底改了哪些欄位
    by_seq: dict = {}
    for row in s.state["change_log"]:
        by_seq.setdefault(row["seq"], []).append(row)

    print(f"\n{'═' * 78}")
    print(f"  {tid}　{who}")
    changed = [r for r in s.state["change_log"]
               if r["seq"] in {c["seq"] for c in calls}]
    print(f"  {len(calls)} 次工具呼叫　{s.state.get('turns', 1)} 個回合"
          f"　{len(changed)} 列資料改動"
          + (f"（另外繼承了上一題的 {inherited} 次呼叫，沒有印出來）"
             if inherited else ""))
    print("═" * 78)

    obs = list(s.state["observations"])

    def show_call(c) -> None:
        args = ", ".join(f"{k}={v!r}" for k, v in (c["args"] or {}).items())
        print(f"  呼叫│ {c['tool']}({args})")
        body = json.dumps(c["data"], ensure_ascii=False) if c["data"] else ""
        if len(body) > cut_at:
            body = body[:cut_at] + f"…（共 {len(body)} 字元，--full 看全部）"
        head = "ok" if c["ok"] else f"擋下來　{c['reason']}"
        print(f"  回覆│ {head}" + (f"　{body}" if body else ""))
        # 把同一筆資料被新增收成一行。新增一筆預約會產生七八個欄位的
        # 改動紀錄，一行一個欄位會把真正重要的改動淹掉。
        groups: dict = {}
        for ch in by_seq.get(c["seq"], []):
            groups.setdefault((ch["table"], ch["key"]), []).append(ch)
        for (table, key), rows in groups.items():
            if all(r["before"] is None for r in rows):
                fields = "、".join(f"{r['field']}={r['after']!r}" for r in rows
                                  if r["after"] is not None
                                  and not str(r["field"]).endswith("_id")
                                  and r["field"] not in ("created_at", "issued_at"))
                print(f"  新增│ {table}.{key}" + (f"　{fields}" if fields else ""))
            else:
                for r in rows:
                    tag = "改動" if r["before"] is not None else "新增"
                    print(f"  {tag}│ {table}.{key}.{r['field']}"
                          f"　{r['before']!r} → {r['after']!r}")

    def show_turn_header(n, at, text, events) -> None:
        print(f"\n─── 回合 {n}　{at}　{clock.weekday(at)} " + "─" * 34)
        for ev in events or []:
            print(f"  事件│ {ev}")
        for ln in str(text).strip().splitlines():
            print(f"  觀察│ {ln}")
        print()

    tr = getattr(agent, "transcript", None) or []
    if tr:
        #  真模型：走它自己的逐字稿，順序就是實際發生的順序
        turn, k = 0, 0
        for row in tr:
            if "observation" in row:
                o = obs[turn] if turn < len(obs) else {}
                turn += 1
                show_turn_header(turn, row["t"], row["observation"],
                                 o.get("events"))
            elif "message" in row:
                msg = row["message"]
                if msg.get("content"):
                    for ln in str(msg["content"]).strip().splitlines():
                        print(f"  模型│ {ln}")
                for tc in msg.get("tool_calls") or []:
                    if tc.get("executed", True):
                        continue
                    # 同一則訊息裡第二個以後的呼叫沒有被執行，
                    # 但要印出來。
                    print(f"  未執行│ {tc.get('name')}"
                          f"（同一則訊息的第二個以後）")
            elif "tool" in row:
                if k < len(calls):
                    show_call(calls[k])
                k += 1
            elif "unparsed_tool_call" in row:
                # 模型想呼叫這支工具，但那個呼叫留在講話內容裡當文字，
                # 環境沒有執行它。
                print(f"  未執行│ {row['unparsed_tool_call']}"
                      f"（寫成文字，剖析器沒抓到）")
            elif "nudge" in row:
                print(f"  環境│ {row['nudge']}")
            elif "backend_error" in row:
                print(f"  ⚠ 後端錯誤│ {row['backend_error']}")
    else:
        #  腳本：沒有「說的話」，就印觀察 ＋ 呼叫
        turn = 0
        pending = list(calls)
        while pending:
            if turn < len(obs):
                o = obs[turn]
                show_turn_header(turn + 1, o["at"], o["text"], o.get("events"))
            turn += 1
            # 這一回合的呼叫：直到（含）end_turn 為止
            while pending:
                c = pending.pop(0)
                show_call(c)
                if c["tool"] == "end_turn":
                    break

    print("\n" + "═" * 78)
    bad = [c for c in calls if not c["ok"]]
    if bad:
        print(f"  被擋下來的呼叫 {len(bad)} 次："
              + "、".join(f"{c['tool']}→{c['reason']}" for c in bad))
    else:
        print("  每一次呼叫都 ok")
    viol = s.state.get("boundary_violations") or []
    if viol:
        # 這是環境的問題不是 Agent 的問題：模擬代理人說了不該說的話。
        print(f"  模擬代理人越線 {len(viol)} 次 —— 這一輪的資料不能用：")
        for v in viol:
            print(f"      {v['at']}　{v['agent_id']}　{v['violation']}")
    for w, label in (("errors", "後端錯誤"), ("unparsed", "呼叫被寫成文字、從未執行"),
                     ("dropped", "多餘呼叫被丟掉")):
        rows = getattr(agent, w, None)
        if rows:
            print(f"  ⚠ {label} {len(rows)} 次")
    print()


#  run

def cmd_run(a) -> int: #用真模型跑一個案例，把逐字稿寫成檔案
    """cmd_run（用真模型跑一個案例，把紀錄寫成檔案）。
      rsandbox run COLO-01/v1_postponed --model <模型名>
      rsandbox run COLO-01/v1_postponed --model <模型名> --repeat 5   同一題跑五次
      rsandbox run all --model <模型名> --repeat 3    所有題目各跑三次

      - 溫度固定 0。亂數種子每一輪往上加一（--seed 是第一輪的種子），
        所以 --repeat 跑的是不同的樣本，不是同一件事算幾遍
      - 演病人的那個模型種子不變，所以變異只來自受測模型
      - 不承諾逐字可重現：本地模型即使溫度 0，仍可能因為批次大小、
        快取、後端版本而有差異，所以每一次都把對話逐字稿寫成檔案
      - 跑完之後用 rsandbox show-run <執行編號> 看逐字稿，或者直接
        rsandbox trace <任務> --model <模型名> 邊跑邊看
      - 這支指令不做任何對錯判定
    """
    from .agents.backends import build_backend
    from .agents.react import ReActAgent
    from .runtime import loop, runio

    d = load_all()
    # 不寫死任務清單。all 的意思是載入了的全部，照代號排序，
    # 所以順序是確定的。
    ids = sorted(d["tasks"]) if a.task == "all" else [a.task]
    rows = []
    for tid in ids:
        task = d["tasks"][tid]
        inh = _inherited(d, task)
        for i in range(1, a.repeat + 1):
            # 每一輪一個不同的種子，所以 --repeat 3 是三個樣本。
            seed = a.seed + i - 1
            backend = build_backend(a.provider, a.model, a.base_url, a.api_key,
                                    seed=seed, temperature=a.temperature,
                                    num_ctx=a.num_ctx or None)
            agent = ReActAgent(backend, max_steps=a.max_steps,
                               tool_calls=a.tool_calls)
            s = loop.run_task(d, task, agent,
                              patient_backend=_patient_backend(a))
            rid = runio.run_id(tid, "react", a.model, i)
            cfg = {"task": tid, "agent": "react", "provider": a.provider,
                   "model": a.model, "seed": seed, "temperature": a.temperature,
                   "max_steps": a.max_steps, "repeat": i,
                   "tool_calls": a.tool_calls,
                   "num_ctx": a.num_ctx or None,
                   # 每一次呼叫的送出量與服務端處理量，見 backends.py。
                   "usage": backend.usage_rows(),
                   "backend_errors": agent.errors,
                   # 被寫成文字、因此從未被執行的工具呼叫，
                   # 見 agents/react.py。
                   "unparsed_tool_calls": agent.unparsed,
                   # 同一則訊息裡第二個以後的呼叫：丟掉，但要計數。
                   "dropped_extra_calls": agent.dropped}
            path = runio.write(rid, cfg, s, agent.transcript)
            rows.append((tid, i, s, agent, path, inh))
            _one_line(tid, i, s, agent, inh)

    if len(rows) > 1:
        _summary(rows)
    return 0


def _ctx_note(usage: list, num_ctx) -> str:
    """_ctx_note（歷史有沒有被服務端截掉），回一段警告或空字串。
      - 判準：送出量超過 context 上限的呼叫有幾次
      - 沒有指定上限就回空字串，不下判定
    """
    if not usage or not num_ctx:
        return ""
    over = sum(1 for u in usage if (u.get("sent_tokens") or 0) > num_ctx)
    return f"  ⚠ {over} 次送出超過 context 上限" if over else ""


def _one_line(tid, i, s, agent, inherited=(0, 0)) -> None: #一次執行印成一行，只有事實沒有判定
    """_one_line（一次執行印成一行）。
    印的東西只有事實：題號、第幾次、幾次呼叫、幾個回合、
    被擋下來幾次，後面再接警告。
    那幾個警告一出現，這一輪的紀錄就不能當成模型行為的觀察。
    只算這一輪的，繼承來的紀錄從筆數裡扣掉。
    """
    n_calls, n_changes = inherited
    calls = s.state["tool_call_log"][n_calls:]
    changes = s.state["change_log"][n_changes:]
    blocked = sum(1 for c in calls if not c["ok"])
    print(f"  {tid:9} r{i}  {len(calls):>3} 次呼叫"
          f"　{s.state.get('turns', 1)} 回合"
          f"　{len(changes):>3} 列改動"
          + (f"　{blocked} 次被擋" if blocked else "")
          + ("  ⚠ 後端錯誤" if agent.errors else "")
          + (f"  ⚠ {len(u)} 個呼叫寫成文字"
             if (u := getattr(agent, "unparsed", None)) else "")
          + (f"  ⚠ {len(dr)} 個多餘呼叫被丟掉"
             if (dr := getattr(agent, "dropped", None)) else "")
          + _ctx_note(agent.backend.usage_rows(),
                      getattr(agent.backend, "num_ctx", None))
          + (f"  模擬代理人越線 {len(v)} 次"
             if (v := s.state.get("boundary_violations")) else ""))


def _summary(rows) -> None: #跑完多次之後的總結，只數次數
    """_summary（跑完多次之後的總結）。
    一次只跑一題一次的時候不印這個。它只數次數，不下任何判斷。
    """
    print("\n" + "─" * 72)
    n = len(rows)
    print(f"  {n} 輪")
    calls = sum(len(s.state["tool_call_log"]) - inh[0]
                for _t, _i, s, _a, _p, inh in rows)
    print(f"  平均 {calls / n:.1f} 次工具呼叫")
    # 下面這一行不是效能指標，是這批紀錄能不能用的指標：模型把工具
    # 呼叫寫成了講話內容裡的 JSON 文字，那個呼叫從來沒被執行。
    # 不補救，只計數並標注出來。
    lost = sum(len(getattr(a_, "unparsed", None) or [])
               for _t, _i, _s, a_, _p, _inh in rows)
    if lost:
        bad = sum(1 for _t, _i, _s, a_, _p, _inh in rows
                  if getattr(a_, "unparsed", None))
        print(f"  ⚠ 解析污染　{lost} 個工具呼叫被寫成文字、從未執行"
              f"（{bad}/{n} 輪）")
        print("     這幾輪的紀錄不能當成模型行為的觀察。")
    inval = sum(1 for _t, _i, s, _a, _p, _inh in rows
                if s.state.get("boundary_violations"))
    if inval:
        print(f"  {inval}/{n} 輪有模擬代理人越線，那幾輪的資料不能用")
    print("  逐字稿寫在 runs/ 底下，用 show-run 看")
    print("─" * 72 + "\n")


#  check-llm

def cmd_check_llm(a) -> int: #確保模型連線、tool_call正常
    """cmd_check_llm（跑真模型之前的前置檢查）。
    問三件事：接得上嗎、支援原生的函式呼叫嗎、看得懂我們的工具定義嗎。

        rsandbox check-llm --model <模型名>
        rsandbox check-llm --model <模型名> --base-url http://其他機器:11434

      - 它走的是沙盒實際會用的那條程式碼，不是手動發一個 HTTP 請求
      - 只送一則觀察與整份工具定義，不跑完整回合，這一步只驗管線
    """
    from .agents.backends import BackendError, build_backend
    from .agents.schema import all_tool_schemas
    from .loaders import prompts

    d = load_all()
    tools = all_tool_schemas(d["tools"])
    backend = build_backend(a.provider, a.model, a.base_url, a.api_key,
                            seed=0, temperature=0.0,
                            num_ctx=getattr(a, "num_ctx", None) or None)
    system = prompts.system_prompt("referral_agent", d["tools"])
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": "Today is 2026-09-01.\n"
                                    "A referral case has been opened for patient P001."},
    ]

    print(f"\n  provider   {a.provider}")
    print(f"  model      {a.model}")
    print(f"  base_url   {backend.base_url}")
    print(f"  工具       {len(tools)} 個（全部列在提示詞裡，不做工具搜尋）")
    print(f"  提示詞     {len(system)} 字元\n")

    try:
        msg = backend.chat(messages, tools)
    except BackendError as e:
        print(f"✗ 連不上或回應看不懂：\n    {e}\n")
        print("  常見原因："
              "\n    · 模型服務沒跑起來"
              "\n    · 模型還沒下載（試 ollama pull " + a.model + "）"
              "\n    · 這個模型不支援工具呼叫"
              "（試 ollama show " + a.model + " 看 Capabilities）\n")
        return 1

    calls = msg.get("tool_calls") or []
    if msg.get("content"):
        print(f"  模型說：{msg['content'][:300]}")
    if not calls:
        print("\n✗ 模型沒有發出任何工具呼叫。")
        print("  這通常是這個模型不支援工具呼叫，不是模型太笨 ——")
        print(f"  跑 `ollama show {a.model}` 看 Capabilities 有沒有 tools。")
        return 1

    print(f"\n✓ 收到 {len(calls)} 個 tool_call：")
    for c in calls:
        print(f"    {c['name']}({json.dumps(c['arguments'], ensure_ascii=False)})")

    known = set(d["tools"])
    bad = [c["name"] for c in calls if c["name"] not in known]
    if bad:
        print(f"\n⚠ 有 {len(bad)} 個不存在的工具名：{bad}")
        print("  這不算失敗 —— 呼叫不存在的工具會收到「參數有問題」，"
              "那本來就是要測的行為之一。")

    first = calls[0]["name"]
    print(f"\n  第一步選了 {first}")
    if first == "get_patient_record":
        print("  （提示詞說 \"Read the patient's record\" —— 它照做了）")
    print("\n✓ 管線通了：連得上、支援原生的工具呼叫、工具定義讀得懂。\n")
    return 0



#  show-prompt

def cmd_show_prompt(a) -> int: #印出 Agent 真正收到的系統提示詞
    """cmd_show_prompt（印出 Agent 真正收到的系統提示詞），一字不差。
      rsandbox show-prompt

      - 這就是模型看到的第一段文字：任務說明 ＋ 每一支工具的簽章
      - 工具清單裡不含實作路徑、不含前置條件，也不含副作用清單
    """
    from .loaders import prompts
    d = load_all()
    txt = prompts.system_prompt("referral_agent", d["tools"],
                                tool_calls=a.tool_calls)
    print("─" * 78)
    print(f"role: system　呼叫模式 {a.tool_calls}")
    print("─" * 78)
    print(txt)
    print("─" * 78)
    print(f"  共 {len(txt)} 字元、{len(txt.splitlines())} 行")
    print("  這段文字每個回合都在對話歷史的最前面，不會被截掉。\n")
    return 0


#  show-raw

def cmd_show_raw(a) -> int: #把一次執行的對話逐字印出來：誰說了什麼、原文長什麼樣。
    """cmd_show_raw（把一次執行的對話逐字印出來）。
      rsandbox show-raw <執行編號>

      - 跟 show-run 的差別：
            show-run  是整理給人讀的摘要
            show-raw  是原始訊息序列，系統、環境、模型、工具四種角色
                      依序排列，不截斷、不重排、不加註
      - 系統提示詞沒有存進檔案（它是固定的、算得出來），這裡即時重建
    """
    from .loaders import prompts
    from .runtime import runio

    folder = runio.RUNS / a.run_id
    tpath = folder / "transcript.jsonl"
    if not tpath.exists():
        print(f"✗ 找不到 {tpath}")
        return 1
    d = load_all()

    def head(role: str, extra: str = "") -> None:
        print(f"\n┏━ {role}{(' ' + extra) if extra else ''} " + "━" * max(0, 68 - len(role) - len(extra)))

    head("system", "（每個回合都在最前面）")
    print(prompts.system_prompt("referral_agent", d["tools"]).rstrip())

    n_turn = 0
    with open(tpath, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if "observation" in row:
                n_turn += 1
                head("user", f"← 環境　回合 {n_turn}　{row['t']}")
                print(row["observation"])
            elif "message" in row:
                m = row["message"]
                calls = m.get("tool_calls") or []
                head("assistant", f"← 模型　{row['t']}")
                c = m.get("content")
                if c:
                    print("content:")
                    print(c)
                else:
                    print("content: (空的 —— 這一步模型一個字都沒講，直接呼叫工具)")
                if calls:
                    print("tool_calls:")
                    for i, tc in enumerate(calls):
                        mark = "執行" if tc.get("executed", i == 0) else "丟棄"
                        print(f"  [{i}] {mark}  {tc.get('name')}"
                              f"({json.dumps(tc.get('arguments'), ensure_ascii=False)})")
                else:
                    print("tool_calls: []   ← 這一則沒有任何原生的工具呼叫")
            elif "unparsed_tool_call" in row:
                print(f"       ⚠ 講話內容裡還有一個 {row['unparsed_tool_call']}，"
                      f"被寫成文字、沒有被執行")
            elif "nudge" in row:
                head("user", "← 環境推了一下（模型只講話、沒呼叫工具）")
                print(row["nudge"])
            elif "tool" in row:
                head("tool", f"← 環境回覆 {row['tool']}")
                print(json.dumps(row["result"], ensure_ascii=False, indent=2))
            elif "backend_error" in row:
                head("error")
                print(row["backend_error"])
    print()
    return 0


#  show-run

def cmd_show_run(a) -> int: #把一次執行的transcript印出
    """cmd_show_run（把一次執行的對話整理成人看得懂的樣子）。
        rsandbox show-run <執行編號>
        rsandbox show-run <執行編號> --full   # 不截斷工具回傳的內容

    讀逐字稿是唯一能看到模型為什麼那樣做的地方。
    """
    from .runtime import runio

    folder = runio.RUNS / a.run_id
    tpath = folder / "transcript.jsonl"
    if not tpath.exists():
        print(f"✗ 找不到 {tpath}（照腳本跑的執行沒有對話逐字稿）")
        return 1
    with open(folder / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)

    cut = 10_000 if a.full else 400
    print(f"\n  {a.run_id}")
    mode = cfg.get("tool_calls", "one")
    print(f"  {cfg.get('model')}　temperature {cfg.get('temperature')}"
          f"　seed {cfg.get('seed')}　呼叫模式 {mode}")
    if usage := cfg.get("usage"):
        # 送出去的量跟服務端實際處理的量。差距大就是歷史被截掉了。
        sent = max((u.get("sent_tokens") or 0) for u in usage)
        done = max((u.get("prompt_eval_count") or 0) for u in usage)
        lim = cfg.get("num_ctx")
        print(f"  context　送出最多 {sent:,} token　服務端最多處理 {done:,}"
              f"　上限 {f'{lim:,}' if lim else '未指定'}")
        if note := _ctx_note(usage, lim):
            print(" " + note.rstrip())
    if cfg.get("backend_errors"):
        print(f"  ⚠ 後端錯誤 {len(cfg['backend_errors'])} 次")
    if cfg.get("dropped_extra_calls"):
        dr = cfg["dropped_extra_calls"]
        why = ("一步只執行一個工具" if mode == "one"
               else "排在 end_turn 後面，回合已經結束")
        print(f"  ⚠ {len(dr)} 個多餘呼叫被丟掉（{why}）："
              + "、".join(sorted({x['name'] for x in dr})))
    if cfg.get("unparsed_tool_calls"):
        u = cfg["unparsed_tool_calls"]
        print(f"  ⚠ {len(u)} 個工具呼叫被寫成文字、從未執行："
              + "、".join(sorted({x['name'] for x in u})))
    print("═" * 78)

    turn = 0
    with open(tpath, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if "observation" in row:
                turn += 1
                print(f"\n┌─ 回合 {turn}　{row['t']} ─────────────────────────")
                for ln in row["observation"].splitlines():
                    print(f"│ 環境：{ln}")
                print("└" + "─" * 46)
            elif "message" in row:
                msg = row["message"]
                if msg.get("content"):
                    for ln in str(msg["content"]).strip().splitlines():
                        print(f"  模型│ {ln}")
                for c in msg.get("tool_calls") or []:
                    args = json.dumps(c.get("arguments"), ensure_ascii=False)
                    # 只有第一個呼叫會被執行（提示詞明文：一步一個
                    # 工具），其餘標成多餘，不然呼叫與結果會對不齊。
                    if c.get("executed", True):
                        print(f"  呼叫│ {c.get('name')}({args})")
                    else:
                        print(f"  多餘│ {c.get('name')}({args}) ← 同一則訊息的第二個以後，未執行")
            elif "tool" in row:
                mark = "✓" if row["ok"] else f"✗ {row['reason']}"
                body = json.dumps(row["result"], ensure_ascii=False)
                if len(body) > cut:
                    body = body[:cut] + f"…（{len(body)} 字元，--full 看全部）"
                print(f"  結果│ {mark}　{body}")
            elif "unparsed_tool_call" in row:
                # 模型想呼叫這支工具，但那個呼叫留在講話內容裡當文字，
                # 環境沒有執行它。
                extra = "（同一則訊息另有原生呼叫）" if row.get("had_native_call") else ""
                print(f"  未執行│ {row['unparsed_tool_call']} "
                      f"← 寫成文字，parser 沒抓到{extra}")
            elif "nudge" in row:
                print(f"  環境│ {row['nudge']}")
            elif "backend_error" in row:
                print(f"  ⚠ 後端錯誤│ {row['backend_error']}")

    print("\n" + "═" * 78)
    print(f"  {len(cfg.get('backend_errors') or [])} 次後端錯誤"
          f"　{len(cfg.get('unparsed_tool_calls') or [])} 個呼叫寫成文字"
          f"　{len(cfg.get('dropped_extra_calls') or [])} 個多餘呼叫被丟掉")
    print()
    return 0


#  main

def _quiet_broken_pipe(fn): #head -80` 截斷時不要噴 traceback。
    """被 `| head -80` 截斷時不要噴 traceback。
    head 讀夠了就關掉管線，Python 這端下一次 print 會丟
    BrokenPipeError。那不是錯誤，是正常的 shell 行為。
    """
    import functools

    @functools.wraps(fn)
    def wrapped(a):
        try:
            return fn(a)
        except BrokenPipeError:
            # 把 stdout 換成 /dev/null 再退出。不換的話，直譯器結束時
            # 會再 flush 一次，又丟一個 BrokenPipeError 出來。
            try:
                sys.stdout.flush()
            except BrokenPipeError:
                pass
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
            return 0
    return wrapped


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="rsandbox")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("load", help="把全部資料載入，印出規模").set_defaults(fn=cmd_load)

    sh = sub.add_parser("show", help="看載入結果")
    sh.add_argument("what", choices=["tools", "rules", "slots", "tasks", "nodes"])
    sh.set_defaults(fn=cmd_show)

    tr = sub.add_parser("trace", help="跑一輪並把過程逐步印出來（不判對錯）")
    tr.add_argument("task",
                    help="變體代號 COLO-01/v1_postponed、"
                         "一整組 COLO-01、或 all")
    tr.add_argument("-t", "--trajectory", default=None,
                    help="變體腳本的 id；省略就跑預設腳本")
    tr.add_argument("--full", action="store_true",
                    help="工具回傳的內容不截斷")
    tr.add_argument("--model", default=None,
                    help="給了就用真模型跑，不給就跑腳本")
    tr.add_argument("--provider", default="ollama", choices=["ollama", "openai"])
    tr.add_argument("--base-url", default=None)
    tr.add_argument("--api-key", default=None)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--temperature", type=float, default=0.0)
    tr.add_argument("--max-steps", type=int, default=100)
    _add_ctx_arg(tr)
    _add_tool_call_arg(tr)
    _add_patient_args(tr)
    tr.set_defaults(fn=cmd_trace)

    r = sub.add_parser("run", help="用真模型跑一個案例")
    r.add_argument("task", help="任務代號，或 all 跑全部")
    r.add_argument("--model", required=True)
    r.add_argument("--provider", default="ollama", choices=["ollama", "openai"])
    r.add_argument("--base-url", default=None)
    r.add_argument("--api-key", default=None)
    r.add_argument("--repeat", type=int, default=1)
    r.add_argument("--seed", type=int, default=0,
                   help="第一輪的亂數種子。第 n 輪用 seed+n-1")
    r.add_argument("--temperature", type=float, default=0.0)
    _add_patient_args(r)
    # max-steps（單回合步數上限）限制的是一個回合裡模型與環境來回幾次，
    # 不是工具呼叫數。模型只講話、不呼叫工具的那一次也會消耗它，但不會
    # 增加步數，所以執行紀錄印的 N 步永遠小於這個上限。
    # tool-calls 是 batch 的時候一次來回可以跑好幾個工具，兩者差更多。
    r.add_argument("--max-steps", type=int, default=100,
                   help="一個回合最多幾次模型來回（防原地打轉）。預設 100")
    _add_ctx_arg(r)
    _add_tool_call_arg(r)
    r.set_defaults(fn=cmd_run)

    cl = sub.add_parser("check-llm", help="檢查模型接得上、支援 function calling")
    cl.add_argument("--model", required=True)
    cl.add_argument("--provider", default="ollama", choices=["ollama", "openai"])
    cl.add_argument("--base-url", default=None)
    cl.add_argument("--api-key", default=None)
    _add_ctx_arg(cl)
    cl.set_defaults(fn=cmd_check_llm)

    shr = sub.add_parser("show-run", help="把一次執行的 transcript 印出來")
    shr.add_argument("run_id")
    shr.add_argument("--full", action="store_true", help="不截斷工具結果")
    shr.set_defaults(fn=cmd_show_run)

    sp = sub.add_parser("show-prompt", help="印出 Agent 收到的系統提示詞")
    _add_tool_call_arg(sp)
    sp.set_defaults(fn=cmd_show_prompt)

    sraw = sub.add_parser("show-raw", help="逐字印出一次執行的原始訊息序列")
    sraw.add_argument("run_id")
    sraw.set_defaults(fn=cmd_show_raw)

    c = sub.add_parser("call", help="手動打一個工具")
    c.add_argument("task")
    c.add_argument("tool")
    c.add_argument("args", nargs="?", default="")
    c.set_defaults(fn=cmd_call)

    a = p.parse_args(argv)
    return _quiet_broken_pipe(a.fn)(a)


if __name__ == "__main__":
    sys.exit(main())
