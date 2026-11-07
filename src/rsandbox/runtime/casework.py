"""哪一筆預約才算數：全系統唯一的定義。
寄說明單、開醫囑、下抽血單都要知道主處置是哪一筆才能抄日期，
所以這個判斷只寫一次，別處不另外寫。
  - 這裡沒有任何臨床判斷。挑的規則純粹是行政性的：日期最早的那一筆，
    同一天再用編號決勝。它不看診次型別對不對，也不看這個處置適不適合病人
  - 重複預約不會被環境擋下來，照樣成功，只會被記成一個獨立的錯誤
"""
from __future__ import annotations

LIVE = ("booked", "done")           # LIVE：還算數的預約狀態（已訂、已完成）


def booked_appointments(state: dict) -> list[dict]:
    """booked_appointments（所有還算數的預約），依「日期、編號」排序。
    排序固定，所以每次跑結果都一樣。
    """
    return sorted((a for a in state["appointments"].values()
                   if a.get("status") in LIVE),
                  key=lambda a: (a.get("date") or "", a["appointment_id"]))


def primary_appointment(state: dict) -> dict | None:
    """primary_appointment（主處置是哪一筆）：日期最早的那一筆已訂預約。
    如果全部都被取消了，就退回去取編號最大的那一筆，這樣約了又取消的
    軌跡還是有東西可以拿來算多停了幾天藥。
    """
    live = booked_appointments(state)
    if live:
        return live[0]
    rows = [a for _, a in sorted(state["appointments"].items())]
    return rows[-1] if rows else None


def duplicate_bookings(state: dict) -> list[str]:
    """duplicate_bookings（哪些項目被重複訂了）。
    以診次型別分組，不是把所有預約混在一起數：一個案子同時有手術、
    復健、牙科好幾筆預約是正常的，同一個型別有兩筆以上才叫重複。
    """
    groups: dict[str, int] = {}
    for a in booked_appointments(state):
        key = a.get("slot_type") or ""
        groups[key] = groups.get(key, 0) + 1
    return sorted(k for k, n in groups.items() if n > 1)


#  醫囑：以「藥」「檢驗」為單位

def norm(name) -> str:
    """norm（normalise，正規化）：藥名、檢驗名去掉前後空白、轉成小寫。
    只做這兩件事，到此為止。
      - Agent 寫 "Warfarin"、參考答案算出 "warfarin"，那只是同一個
        名字的大小寫差別，不是臨床判斷，所以可以對齊
      - warfarin sodium 算不算 warfarin 是臨床判斷，這裡不做。模型寫出
        別的寫法，查到的就是這個藥沒有有效醫囑
    """
    return str(name or "").strip().lower()


def active_holds_by_drug(state: dict) -> dict[str, list[dict]]:
    """active_holds_by_drug（按藥名分組的有效停藥醫囑）。
    按藥分組：一個病人同時要停兩種藥的時候，兩張醫囑都有效才是正確的，
    所以要同一種藥有兩張以上才算沒收好尾。
    """
    out: dict[str, list[dict]] = {}
    for _, o in sorted(state["hold_orders"].items()):
        if o.get("status") == "active":
            out.setdefault(norm(o["drug"]), []).append(o)
    return out


def active_labs_by_test(state: dict) -> dict[str, list[dict]]:
    """active_labs_by_test（按檢驗項目分組的有效抽血單）。理由同上。"""
    out: dict[str, list[dict]] = {}
    for _, o in sorted(state["lab_orders"].items()):
        if o.get("status") == "active":
            out.setdefault(norm(o["test"]), []).append(o)
    return out
