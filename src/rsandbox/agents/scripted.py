"""照腳本執行的假 Agent，不是被測對象。
  - 它照一份寫好的步驟清單一步一步呼叫工具，不做任何判斷，
    用途是在不接模型的情況下把一個任務從頭跑到尾
  - 軌跡（一串工具呼叫）寫在任務檔裡，不寫在程式裡
  - 介面跟真模型那一邊完全一樣：

        activate(session, observation) -> None    呼叫一次 ＝ 一個回合

  - 一個回合會消耗腳本到下一個 end_turn（含）為止
  - 時鐘與事件由主迴圈管，這裡不碰

佔位符（placeholder）：後面的步驟常常要引用前面產生的代號。
腳本裡寫 {{HO}}、{{PREP}}、{{APT}}、{{LAB}}，意思是那張表目前
最新的那一筆。
"""
from __future__ import annotations

import copy
import re

PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")

# LATEST：佔位符 -> (要去哪張表找, 那張表的代號欄位)
LATEST = {
    "HO": ("hold_orders", "order_id"),
    "LAB": ("lab_orders", "lab_order_id"),
    "PREP": ("prep_instructions", "instruction_id"),
    "APT": ("appointments", "appointment_id"),
}


def _latest_id(state: dict, kind: str) -> str:
    """_latest_id（某張表目前最新那一筆的代號）。"""
    table, _field = LATEST[kind]
    rows = state[table]
    if not rows:
        raise ValueError(f"軌跡引用了 {{{{{kind}}}}}，但 {table} 是空的")
    return sorted(rows)[-1]


def _fill(value, state: dict):
    """_fill（把參數裡的佔位符換成真正的代號），會往巢狀結構裡遞迴。"""
    if isinstance(value, str):
        return PLACEHOLDER.sub(lambda m: _latest_id(state, m.group(1)), value)
    if isinstance(value, list):
        return [_fill(v, state) for v in value]
    if isinstance(value, dict):
        return {k: _fill(v, state) for k, v in value.items()}
    return value


class ScriptedAgent:
    """ScriptedAgent（照腳本跑的 Agent）。"""

    name = "scripted"

    def __init__(self, steps: list[dict]) -> None:
        self.steps = copy.deepcopy(steps)
        self.i = 0
        self.results: list[dict] = []
        self.observations: list[str] = []

    # ---------------- 主迴圈的介面 ----------------

    def activate(self, session, observation: str) -> None:
        """activate（跑一個回合）：消耗腳本到下一個 end_turn（含）為止。
        腳本裡沒寫 end_turn 就一路跑到底，失敗的呼叫不中斷。
        """
        self.observations.append(observation)
        while self.i < len(self.steps):
            step = self.steps[self.i]
            self.i += 1
            args = _fill(step.get("args") or {}, session.state)
            env = session.call(step["tool"], args)
            self.results.append({"tool": step["tool"], "args": args, **env})
            if step["tool"] == "end_turn" and env["ok"]:
                return

    def finished(self) -> bool:
        """finished（腳本跑完了沒有）。
        主迴圈會先問這個方法再決定要不要結束。真模型沒有這個方法，
        主迴圈對它改用這回合有沒有寫入去推（見 runtime/loop.py）。
        """
        return self.i >= len(self.steps)

    # ---------------- 檢查 ----------------

    def failures(self) -> list[dict]:
        """failures（這次跑下來有哪些呼叫失敗了）。"""
        return [r for r in self.results if not r["ok"]]

    def run(self, session) -> list[dict]:
        """run（把整份腳本一次跑完），不經過主迴圈。
          - 它會在每次 end_turn 自己推進時鐘，因為 v0 的流程跨好幾天
          - 要跑有擾動事件的案例請用 runtime.loop.run_task()，
            只有那邊會觸發事件
        """
        while not self.finished():
            self.activate(session, "")
            if self.finished():
                break
            clock = session.task["clock"]
            if session.today < clock["end_date"]:
                session.advance(int(clock.get("days_per_turn", 1)))
        return self.results
