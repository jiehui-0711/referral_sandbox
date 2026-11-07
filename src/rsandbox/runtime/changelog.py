"""變更紀錄：只追加、不修改的狀態異動流水帳。
  - 它回答某一天當時的狀態是什麼，例如擾動發生之前 Agent 選的是哪一家
    醫院、舊的那張停藥醫囑是哪一天被取代的。最終狀態只有現在，
    回答不了這些
  - 狀態照舊放在記憶體裡，不用資料庫：要的是每一次改變都留下痕跡，
    而資料庫的約束正是第一條不變量明文禁止的，停藥日期那一欄必須是
    完全沒有約束的欄位
  - 產生的方式：在 Session.call() 裡，handler 執行前後各拍一次快照，
    把差異攤平成一列一列「哪張表、哪一筆、哪一欄、從什麼變成什麼」。
    放在那裡而不是放在每一支 handler 裡，所以 handler 不知道它存在，
    新增工具也不用改這裡
  - Agent 讀不到它。查案件狀態回的是目前的狀態，不是歷史；
    沒有任何工具會回傳這張表
  - 它是紀錄，不是約束：只記下已經發生的事，不阻止任何事發生
"""
from __future__ import annotations

import copy

# TRACKED（要追蹤的表）。工具會寫到的世界狀態全部列在這裡。
TRACKED = ("assessments", "appointments", "hold_orders", "lab_orders",
           "prep_instructions", "contact_log", "referrals", "escalations",
           "series", "summaries")

# NOT_WORLD（在狀態裡但不追蹤的表）。
# 這份清單與 TRACKED 合起來要蓋住狀態裡的每一張表，兩邊都沒有的
# 由 checks/tests/test_changelog.py 指名。
#   - tool_call_log、change_log：追蹤自己會無限遞迴
#   - observations、boundary_violations：旁路的紀錄
#   - position：Agent 站的位置
#   - sim_clock：病人模擬的計時
NOT_WORLD = ("tool_call_log", "change_log", "observations",
             "boundary_violations", "position", "sim_clock", "turns")


def _flatten(rows: dict, table: str) -> dict:
    """_flatten（攤平）：把巢狀的資料壓成「(表, 列, 欄) -> 值」的平面字典。
    裡面本來就是一整塊的值（例如說明單的快照）就整塊當一個值，不再往下拆。
    """
    out: dict[tuple[str, str, str], object] = {}
    for key, row in (rows or {}).items():
        if not isinstance(row, dict):
            out[(table, str(key), "")] = row
            continue
        for field, value in row.items():
            out[(table, str(key), field)] = value
    return out


def capture(state: dict, records: dict, slots: dict) -> dict:
    """capture（拍一張快照）：把目前狀態壓成可以直接比對的平面字典。"""
    snap: dict = {}
    for table in TRACKED:
        snap.update(_flatten(state.get(table), table))
    for pid, rec in (records or {}).items():
        # 病歷目前只有電話會被工具改到，但這裡比較全部欄位，
        # 哪天多一支會改病歷的工具不用跟著改。
        for field, value in rec.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                snap[("records", pid, field)] = value
    for sid, slot in (slots or {}).items():
        snap[("slots", sid, "booked_count")] = slot.get("booked_count")
    return snap


def diff(before: dict, after: dict) -> list[dict]:
    """diff（差異）：比對前後兩張快照，一欄一欄列出哪裡變了。
    依「表、列、欄」排序，所以同一次執行每次跑出來的順序都一樣。
    """
    rows = []
    for key in sorted(before.keys() | after.keys()):
        old = before.get(key)
        new = after.get(key)
        if old == new:
            continue
        table, row_key, field = key
        rows.append({"table": table, "key": row_key, "field": field,
                     "before": copy.deepcopy(old), "after": copy.deepcopy(new)})
    return rows


def state_as_of(change_log: list[dict], day: str) -> dict:
    """state_as_of（某一天當時的狀態）：重建「那天結束時」長什麼樣。
    只含被改過的欄位：每一欄取它在那天（含當天）之前最後一次變成的值。
    用途是回答擾動發生前 Agent 選的是哪一家這類問題。
    """
    out: dict[tuple[str, str, str], object] = {}
    for rec in change_log:
        if rec["at"] > day:
            continue
        out[(rec["table"], rec["key"], rec["field"])] = rec["after"]
    return out
