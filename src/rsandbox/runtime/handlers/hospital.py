"""預約：訂號、訂一整組療程、改期、取消，以及術前抽血單。
這一組工具只動預約本身。改期只改一個日期欄位，以下它一個都沒動：
  - 停藥醫囑
  - 抽血單
  - 病人手上的說明單
  - 對方醫院手上的轉診單

這件事只寫在註解裡，不寫進工具描述。
"""
from __future__ import annotations

from .. import casework
from ..clock import weekday
from ..envelope import ToolError
from ..messages import render


def _slot(ctx, sid: str) -> dict:
    """_slot（照代號把一個診次撈出來）。"""
    return ctx.world["slots"][sid]


# NOTIFIES（每一支會改動預約的工具，系統要不要自動通知病人）。
#   - 值是 messages.py 的模板名，None ＝ 不發
#   - 每一支會改動預約的工具都要在這張表上，缺的由
#     checks/tests/test_changelog.py 指名
#   - 發出去的通知只帶日期，不帶用藥、抽血或任何依附在手術日上的欄位
NOTIFIES = {
    "reschedule_appointment": "appointment_moved",
    "book_appointment": None,
    "cancel_appointment": None,
}


def _notify_patient(ctx, template: str, **fields) -> None:
    """_notify_patient（系統主動發一則通知給病人），寫進 contact_log。
    單方面發出，沒有回覆；Agent 問病人答的那兩支在 clinic.py。
    """
    cid = ctx.new_id("contact_log")
    ctx.state["contact_log"][cid] = {
        "log_id": cid, "direction": "outbound",
        "content": render(template, **fields),
        "created_at": ctx.today,
    }


def notify_on_change(ctx, tool_id: str, **fields) -> None:
    """notify_on_change（照 NOTIFIES 決定要不要發通知）。
    不在表上、或值是 None，就什麼都不做。
    """
    template = NOTIFIES.get(tool_id)
    if template:
        _notify_patient(ctx, template, **fields)


def _appt_view(ap: dict) -> dict:
    """_appt_view（一筆預約回給 Agent 的樣子）：只挑它該看到的欄位。"""
    return {k: ap[k] for k in ("appointment_id", "slot_id", "institution_id",
                               "dept", "service", "slot_type", "date",
                               "period", "status")}


def _write_appointment(ctx, slot: dict, series_id: str | None = None) -> dict:
    """_write_appointment（把一個診次寫成一筆預約）。
    機構、科別、項目、診次型別、日期這幾樣是抄寫進預約裡的，不是存一個
    指標再回頭查。所以診次資料改了，這筆預約上的值不會跟著變。
    """
    slot["booked_count"] += 1
    aid = ctx.new_id("appointments")
    ap = {
        "appointment_id": aid,
        "slot_id": slot["slot_id"],
        "institution_id": slot["institution_id"],
        "dept_id": slot["dept_id"],
        "dept": slot["dept"],
        "service": slot["service"],
        "slot_type": slot["slot_type"],
        "date": slot["date"],
        "period": slot["period"],
        "series_id": series_id,
        "status": "booked",
        "booked_at": ctx.today,
        "history": [],
    }
    ctx.state["appointments"][aid] = ap
    return ap


def _release(ctx, ap: dict) -> None:
    """_release（把一筆預約佔走的位置還回去）。"""
    slot = ctx.world["slots"].get(ap["slot_id"])
    if slot is not None and slot["booked_count"] > 0:
        slot["booked_count"] -= 1


#  預約

def book_appointment(ctx):
    """book_appointment（訂一個診次）。
    行政關卡（有沒有位置、日期過了沒、要不要先有轉診）在前一關就檢查完了。
    """
    return _appt_view(_write_appointment(ctx, _slot(ctx, ctx.args["slot_id"])))


def book_series(ctx):
    """book_series（一次把一整組療程訂完）。
      - 復健是「每週兩次、做六週、共十二次」這種形式，一次訂完
      - 整組訂，或整組不訂。寫入先全部算好再一次落地
    """
    picks = series_slots(ctx)
    sid = ctx.new_id("series")
    ctx.state["series"][sid] = {
        "series_id": sid,
        "institution_id": ctx.args["institution_id"],
        "dept": ctx.args["dept"],
        "service": ctx.args["service"],
        "weekdays": list(ctx.args["weekdays"]),
        "count": int(ctx.args["count"]),
        "status": "booked",
        "booked_at": ctx.today,
    }
    dates = [_write_appointment(ctx, s, series_id=sid)["date"] for s in picks]
    return {"series_id": sid, "dates": dates}


def series_slots(ctx) -> list[dict]:
    """series_slots（這一組療程會落在哪幾個診次）。
    前置條件與實際寫入共用這同一份計算，所以不會出現前置條件說排得下、
    寫入的時候卻排出別的東西。
    """
    dept_id = f"{ctx.args['institution_id']}_{ctx.args['dept']}"
    want = set(ctx.args["weekdays"])
    out: list[dict] = []
    for sid, s in sorted(ctx.world["slots"].items(),
                         key=lambda kv: (kv[1]["date"], kv[0])):
        if len(out) >= int(ctx.args["count"]):
            break
        if s["dept_id"] != dept_id or s["service"] != ctx.args["service"]:
            continue
        if s["date"] < ctx.args["first_date"] or s["status"] != "open":
            continue
        if weekday(s["date"]) not in want:
            continue
        if s["booked_count"] >= s["capacity"]:
            continue
        out.append(s)
    return out


def reschedule_appointment(ctx):
    """reschedule_appointment（把一筆預約移到另一個診次）。
      - 只改一個日期欄位。停藥醫囑、抽血單、說明單、轉診單都沒動
      - 這件事只寫在註解裡，不寫進工具描述

    寫入的順序：先把舊位置還回去，再去佔新位置。佔新位置失敗時舊位置
    要還原，不然會留下一個「誰都沒佔但顯示已滿」的鬼位置。
    """
    ap = ctx.state["appointments"][ctx.args["appointment_id"]]
    old, new = _slot(ctx, ap["slot_id"]), _slot(ctx, ctx.args["new_slot_id"])

    old["booked_count"] -= 1
    try:
        if new["booked_count"] >= new["capacity"]:
            raise ToolError("capacity_full", "新診次在寫入前被填滿")
        new["booked_count"] += 1
    except Exception:
        old["booked_count"] += 1          # 還原
        raise

    ap["history"].append({"slot_id": ap["slot_id"], "date": ap["date"]})
    for k in ("slot_id", "institution_id", "dept_id", "dept", "service",
              "slot_type", "date", "period"):
        ap[k] = new[k]

    notify_on_change(ctx, "reschedule_appointment", date=new["date"])
    return _appt_view(ap)


def cancel_appointment(ctx):
    """cancel_appointment（取消一筆預約，或一整組療程還沒到的場次）。
      - 不會連帶撤銷任何醫囑。手術取消了、停藥醫囑還生效著，
        病人就照舊停藥
      - 已經發生過的場次不動
    """
    out = []
    if ctx.args.get("series_id"):
        ser = ctx.state["series"][ctx.args["series_id"]]
        ser["status"] = "cancelled"
        ser["cancelled_at"] = ctx.today
        for aid, ap in sorted(ctx.state["appointments"].items()):
            if ap.get("series_id") != ser["series_id"] or ap["status"] != "booked":
                continue
            if ap["date"] <= ctx.today:
                continue              # 已經發生的不動
            _release(ctx, ap)
            ap["status"] = "cancelled"
            ap["cancelled_at"] = ctx.today
            out.append(aid)
    else:
        ap = ctx.state["appointments"][ctx.args["appointment_id"]]
        _release(ctx, ap)
        ap["status"] = "cancelled"
        ap["cancelled_at"] = ctx.today
        out.append(ap["appointment_id"])
    return {"cancelled": out}


#  術前抽血

def order_preop_lab(ctx):
    """order_preop_lab（開立術前抽血單）。
      - 抽血日期與抽血地點完全不被驗證：不驗跟預約日的關係、不驗是不是
        已經過去了、不驗那家醫院收不收外院報告。跟停藥醫囑的末次服藥日
        一樣，是完全沒有約束的欄位
      - 三個衍生值的偏移量不同：停藥是處置日往前五天，抽血是往前一天，
        而地點根本不是日期
      - 只有明寫「要取代哪一張」才會取代舊單，規則跟停藥醫囑一致。不寫
        就會留下兩張生效中的單子，環境不阻止
    """
    prev = ctx.args.get("supersedes")
    if prev is not None:
        ctx.state["lab_orders"][prev]["status"] = "superseded"
    oid = ctx.new_id("lab_orders")
    ctx.state["lab_orders"][oid] = {
        "lab_order_id": oid,
        "test": ctx.args["test"],
        "draw_date": ctx.args["draw_date"],
        "site": ctx.args["site"],
        "status": "active",
        "supersedes": prev,
        "issued_at": ctx.today,
    }
    return {"lab_order_id": oid, "status": "active"}
