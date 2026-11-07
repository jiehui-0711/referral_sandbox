"""編號產生器：每一筆新資料的代號怎麼來的。
兩次執行必須逐字相同，所以編號不能用亂數也不能用時間戳，
只能用「每一種前綴各自數自己的流水號」。

    A-001     評估紀錄（assessment）
    APT-001   預約（appointment）
    HO-001    停藥醫囑（hold order）
    LAB-001   抽血單（lab order）
    PREP-001  準備說明單（prep instruction）
    CL-001    聯絡紀錄（contact log）
    REF-001   轉診單（referral）
    SER-001   一整組療程（series）
    ESC-001   上報醫師（escalation）
    SUM-001   寄給病人的方案整理（summary）
"""
from __future__ import annotations

# PREFIX：資料表名稱 -> 這張表的編號前綴
PREFIX = {
    "assessments": "A",
    "appointments": "APT",
    "hold_orders": "HO",
    "lab_orders": "LAB",
    "prep_instructions": "PREP",
    "contact_log": "CL",
    "referrals": "REF",
    "series": "SER",
    "escalations": "ESC",
    "summaries": "SUM",
}


class IdMaker:
    """IdMaker（編號產生器）。一個 Session 一台，各自數各自的。"""

    def __init__(self) -> None:
        self._n: dict[str, int] = {}

    def next(self, prefix: str) -> str:
        """next（給我下一個編號），例如 APT-003。"""
        self._n[prefix] = self._n.get(prefix, 0) + 1
        return f"{prefix}-{self._n[prefix]:03d}"

    def for_table(self, table: str) -> str:
        """for_table（這張表的下一個編號）。"""
        return self.next(PREFIX[table])

    def seed_from(self, state: dict) -> None:
        """seed_from（從已有的資料把流水號接下去）。
        v1～v5 的起始狀態是 v0 跑完的狀態，裡面已經有 APT-001、
        HO-001 了，不接下去會撞號。
        """
        for table, prefix in PREFIX.items():
            rows = state.get(table) or {}
            hi = 0
            for key in rows:
                tail = str(key).rsplit("-", 1)[-1]
                if tail.isdigit():
                    hi = max(hi, int(tail))
            if hi:
                self._n[prefix] = max(self._n.get(prefix, 0), hi)
