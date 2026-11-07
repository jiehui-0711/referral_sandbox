"""每個案例從什麼狀態開始。任務檔的 initial_state 欄位決定。
    from: empty    從全空的狀態開始
    from: v0_end   沿用本組「無擾動」變體跑完之後的狀態

  - v0_end 是相對於這一組的，不是寫死某個任務的名字。哪一個變體算
    無擾動，是看它自己宣告 from: empty，不是看它的名字叫不叫 clean
  - 跑完之後的狀態不是寫死在檔案裡的，是真的跑一遍那一組的
    v0 腳本再拍快照
  - 要跑 v0 就需要它的腳本。腳本是選填的，沒有腳本的任務不能當
    別人的起點，載入時會明確報錯
"""
from __future__ import annotations

import copy

from .session import Session


def _set_path(root: dict, path: str, value) -> None:
    """_set_path（照一條路徑把值寫進多層的資料裡）。
    任務檔裡寫的是 "slots.H1-GI-COL-20260910-AM.status" 這種用點號
    串起來的路徑，意思是 slots 底下那一筆診次的 status 欄位。
    路徑寫錯會當場報錯，不會默默新增一個不存在的欄位。
    """
    keys = path.split(".")
    cur = root
    for k in keys[:-1]:
        if k not in cur:
            raise ValueError(f"覆寫路徑不存在：{path}（斷在 {k!r}）")
        cur = cur[k]
    if keys[-1] not in cur:
        raise ValueError(f"覆寫路徑不存在：{path}（斷在 {keys[-1]!r}）")
    cur[keys[-1]] = value


def clean_end(data: dict, case_id: str) -> dict:
    """clean_end（某一組的「無擾動」變體跑完之後的狀態快照）。
    腳本裡只要有一次呼叫失敗，就直接報錯，不帶著壞掉的狀態往下跑。
    """
    from ..agents.scripted import ScriptedAgent
    from ..loaders.cases import clean_variant

    tid = clean_variant(data["tasks"], case_id)
    steps = (data.get("scripts") or {}).get(tid, {}).get("reference")
    if not steps:
        raise ValueError(
            f"{tid} 沒有腳本，所以它不能當別的變體的起點。"
            f"要嘛補一份 scripts/{tid.split('/')[-1]}.yaml，"
            f"要嘛把接它的變體改成 from: empty")
    base = data["tasks"][tid]
    s = Session(data, base)
    agent = ScriptedAgent(steps)
    agent.run(s)
    bad = agent.failures()
    if bad:
        raise RuntimeError(
            f"{tid} 的腳本有失敗的呼叫，起始狀態不可信：\n"
            + "\n".join(f"  {r['tool']} -> {r['reason']}" for r in bad))
    return s.snapshot()


def build(data: dict, task: dict, base_snapshot: dict | None = None,
          patient_backend=None) -> Session:
    """build（組出一個準備好的 Session）。
    做三件事：載入起始狀態、套用任務檔指定的覆寫、把時鐘設到開始日。

    覆寫（overrides）分兩種，都寫在任務檔裡：
        world_overrides    改世界（例如把某個診次設成停診）
        records_overrides  改病歷

    病人的個人資料不走 overrides，走變體自己的 patient_doc_patch，
    它補的是散文段落，用路徑覆寫表達不了。
    """
    src = (task.get("initial_state") or {}).get("from", "empty")
    if src == "empty":
        s = Session(data, task, patient_backend=patient_backend)
    elif src == "v0_end":
        snap = copy.deepcopy(base_snapshot
                             or clean_end(data, task["case_id"]))
        s = Session(data, task, state=snap["state"],
                    today=task["clock"]["start_date"],
                    patient_backend=patient_backend)
        s.records = snap["records"]
        s.world["slots"] = snap["slots"]
        s.ids.seed_from(s.state)
    else:
        raise ValueError(f"不認得的 initial_state.from：{src!r}")

    ini = task.get("initial_state") or {}
    for ov in ini.get("world_overrides") or []:
        _set_path({"slots": s.world["slots"], **s.state}, ov["path"], ov["value"])
    for ov in ini.get("records_overrides") or []:
        _set_path(s.records, ov["path"], ov["value"])

    # 病人的個人資料：補新的段落，或蓋掉同名的段落。
    patch = task.get("patient_doc_patch") or {}
    for name, text in (patch.get("knows") or {}).items():
        s.patient_doc.setdefault("knows", {})[name] = text
    for e in patch.get("log") or []:
        s.patient_doc.setdefault("log", []).append(dict(e))

    # 起始狀態裡的呼叫紀錄是 v0 留下的，要保留：病人手上那張單子是
    # 哪一天寄的要靠它才查得到。每一筆都帶著日期，所以分得出哪些是
    # v0 做的、哪些是這一題做的。
    s.state["sim_clock"]["today"] = s.today
    return s
