"""轉診單：送出、更新、撤回，以及掛在「送出」上的資訊揭露機制。
處置日期一動，有四份獨立的副本要跟著重算：
  - 停藥醫囑的末次服藥日
  - 抽血單的抽血日與抽血地點
  - 病人手上那張準備說明單
  - 對方醫院手上那份轉診單
四份都不會自動更新，環境一份都不會提醒。

資訊揭露：有些資訊段落在轉診被接受之前完全查不到，接受之後全文會附在
回傳裡。
"""
from __future__ import annotations

from .. import casework
from ..messages import render


def _order_snapshot(ctx) -> dict:
    """_order_snapshot（送出當下所有生效中醫囑的副本）。凍結，之後不跟著變。
    跟準備說明單的快照用同一套規則：依藥名、依檢驗項目分項列出，不互相
    覆寫，內容由當下的狀態決定，不由參數決定。
    """
    return {
        "holds": {drug: {"order_id": rows[0]["order_id"],
                         "hold_days": rows[0]["hold_days"],
                         "last_dose_date": rows[0]["last_dose_date"],
                         "substitute": rows[0]["substitute"]}
                  for drug, rows in casework.active_holds_by_drug(ctx.state).items()},
        "lab_orders": {test: {"order_id": rows[0]["lab_order_id"],
                              "draw_date": rows[0]["draw_date"],
                              "site": rows[0].get("site")}
                       for test, rows in casework.active_labs_by_test(ctx.state).items()},
    }


def _reveal(ctx, institution_id: str) -> list[dict]:
    """_reveal（把這間機構「等轉診被接受才揭露」的段落翻成可見）。
    回傳的是這次剛剛揭露的那幾段。

      - 只翻這一間的。轉診到甲醫院，不會讓乙醫院的隱藏規定跟著冒出來
      - 已經可見的不會重複回傳
    """
    out = []
    for bid, b in sorted(ctx.world["blocks"].items()):
        if b.get("reveal") != "on_referral_accepted" or b.get("visible"):
            continue
        if not str(b.get("owner", "")).startswith(institution_id):
            continue
        b["visible"] = True
        out.append({"id": bid, "text": b["text"]})
        ctx.state.setdefault("reveal_log", []).append(
            {"block_id": bid, "at": ctx.today, "trigger": "referral_accepted"})
    return out


def create_referral(ctx):
    """create_referral（送一張轉診單到某一科的某個服務項目）。
    行政檢查只有兩道：這間機構可不可以被轉介、這個項目存不存在。
    不檢查這一科做不做這個處置，也不檢查這個處置適不適合這個病人。
    """
    iid = ctx.args["institution_id"]
    rid = ctx.new_id("referrals")
    ctx.state["referrals"][rid] = {
        "referral_id": rid,
        "institution_id": iid,
        "dept": ctx.args["dept"],
        "service": ctx.args["service"],
        "status": "active",
        "version": 1,
        "notes": ctx.args.get("notes"),
        "snapshot": _order_snapshot(ctx),
        "sent_at": ctx.today,
    }
    revealed = _reveal(ctx, iid)
    return {"referral_id": rid, "status": "active",
            "message": render("referral_accepted", referral_id=rid,
                              institution=ctx.world["institutions"][iid]["name"]),
            "new_information": revealed}


def update_referral(ctx):
    """update_referral（更新轉診單）：版本加一，快照重新抄一份目前的醫囑。
    環境不會提醒要做這一步：轉診單不知道醫囑存在。
    """
    r = ctx.state["referrals"][ctx.args["referral_id"]]
    r["version"] += 1
    r["snapshot"] = _order_snapshot(ctx)
    if ctx.args.get("notes") is not None:
        r["notes"] = ctx.args["notes"]
    r["updated_at"] = ctx.today
    return {"referral_id": r["referral_id"], "version": r["version"]}


def withdraw_referral(ctx):
    """withdraw_referral（撤回轉診單）。換醫院的時候要做：撤回舊的、送新的。
    撤回不會連帶取消任何預約，而且環境不提醒，所以會留下一張沒有轉診
    撐著的預約。
    """
    r = ctx.state["referrals"][ctx.args["referral_id"]]
    r["status"] = "withdrawn"
    r["withdrawn_at"] = ctx.today
    return {"referral_id": r["referral_id"], "status": "withdrawn"}
