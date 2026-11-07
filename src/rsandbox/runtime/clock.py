"""模擬時鐘：日期的加減與星期幾。
日期一律用 "2026-09-10" 這種字串表示。
這麼寫有個好處：字串的大小順序剛好等於時間的先後順序，
所以比大小、排序都可以直接比字串，不用先轉成日期物件。

時鐘只在 Agent 呼叫 end_turn（結束回合）時前進，
每次前進幾天寫在任務檔的 clock 裡。
"""
from __future__ import annotations

import datetime as _dt


def parse(d: str) -> _dt.date:
    """parse（把日期字串轉成日期物件），只在需要做加減時才用。"""
    return _dt.date.fromisoformat(d)


def shift(d: str, days: int) -> str:
    """shift（某一天往後／往前幾天），回傳一樣格式的字串。"""
    return (parse(d) + _dt.timedelta(days=days)).isoformat()


WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def weekday(d: str) -> str:
    """weekday（這一天是星期幾）。"2026-09-10" -> "Thu"。
    醫院的規定全部寫成週二、四上午，所以日期工具一定要回星期幾。
    """
    return WEEKDAYS[parse(d).weekday()]


def diff(a: str, b: str) -> int:
    """diff（兩個日期差幾天）：a 減 b。a 比較晚就是正數。"""
    return (parse(a) - parse(b)).days


def days_between(a: str, b: str) -> list[str]:
    """days_between（從 a 到 b 的每一天），含頭含尾。
    b 比 a 早的話回空清單。
    """
    out, cur = [], parse(a)
    end = parse(b)
    while cur <= end:
        out.append(cur.isoformat())
        cur += _dt.timedelta(days=1)
    return out
