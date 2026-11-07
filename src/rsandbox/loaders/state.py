"""載入執行期資料表的格式定義（state/schema.yaml），並提供空白起始狀態。
schema（結構定義）：哪張表有哪些欄位、每個欄位是什麼型別
工具寫入資料時由 executor 照它檢查型別。
"""
from __future__ import annotations

from pathlib import Path

from ..paths import DATA, load_yaml


def load_schema(root: Path = DATA) -> dict:
    """load_schema（把格式定義檔讀進來）。"""
    return load_yaml(root / "state" / "schema.yaml")


def empty_state() -> dict:
    """empty_state（一個案例開始時的執行期狀態）：所有表都是空的。
    字典（{}）的那幾張是照代號查得到的表，
    清單（[]）的那幾張是照時間一筆一筆往後加的流水帳。
    """
    return {"assessments": {}, "appointments": {}, "hold_orders": {},
            "lab_orders": {}, "prep_instructions": {}, "contact_log": {},
            "referrals": {}, "series": {}, "escalations": {}, "summaries": {},
            "tool_call_log": [], "change_log": [],
            "boundary_violations": [],
            "observations": [],
            # Agent 現在站在哪個節點。每天開始時重設回起始節點。
            "position": {"node": None},
            "sim_clock": {"today": None}}
