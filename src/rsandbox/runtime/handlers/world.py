"""世界探索：移動、三層資訊、查號源、查交通、算日期。
醫院的規定不是結構化欄位，而是一段英文公告。負擔藏在三層，每一層都要查：
  - 機構層：術前抽血必須在本院做，不收外院報告
  - 科別層：大腸鏡要先看過初診才能排
  - 項目層：清腸藥要本人到本院藥局領
回傳裡不會出現角色、揭露時機、變體限定這些欄位。
"""
from __future__ import annotations

from ..clock import diff, shift, weekday
from ..envelope import ToolError
from ..messages import render


def _public_block(world: dict, bid: str) -> dict | None:
    """_public_block（一個資訊段落的公開版本）：只有代號與文字。
    必須是目前可見的才回。有些段落要等轉診被接受之後才揭露。
    """
    b = world["blocks"].get(bid)
    if b is None or not b.get("visible"):
        return None
    return {"id": b["id"], "text": b["text"]}


def _blocks(world: dict, ids) -> list[dict]:
    """_blocks（一串段落代號 -> 它們的公開版本），看不到的直接濾掉。"""
    out = [_public_block(world, bid) for bid in (ids or [])]
    return [b for b in out if b is not None]


#  移動

def move_to(ctx):
    """move_to（移動到相鄰的節點）。
      - 相鄰規則：機構之間全部互通，科別只連到自己所屬的那間醫院
      - 走路的成本只有步數，而步數是充足的
    """
    target = ctx.args["node"]
    ctx.state["position"]["node"] = target
    n = ctx.session.nodes[target]
    return {"node": target, "name": n.get("name"), "type": n["type"],
            "neighbours": list(n["neighbours"])}


#  三層資訊

def list_institutions(ctx):
    """list_institutions（列出可以轉介過去的機構）。
      - 只列可被轉介的。Agent 自己的診所不會出現在裡面
      - 回傳不含任何資訊段落：想知道一家醫院的事，要走過去查
    """
    out = []
    for iid, inst in sorted(ctx.world["institutions"].items()):
        if not inst.get("referable"):
            continue
        out.append({"institution_id": iid, "name": inst["name"],
                    "kind": inst.get("kind")})
    return {"institutions": out}


def get_institution_info(ctx):
    """get_institution_info（查一間機構的資訊，附帶它有哪些科）。
    機構這一層藏的是費用與抽血地點，例如術前抽血須在本院、不收外院報告。
    """
    iid = ctx.args["institution_id"]
    inst = ctx.world["institutions"][iid]
    depts = [{"dept": d["dept"], "name": d.get("name")}
             for _, d in sorted(ctx.world["departments"].items())
             if d["institution_id"] == iid]
    return {"institution_id": iid, "name": inst["name"],
            "kind": inst.get("kind"),
            "information": _blocks(ctx.world, inst.get("block_ids")),
            "departments": depts}


def _dept(ctx):
    """_dept（從參數的機構＋科別組出完整代號，把那一科撈出來）。"""
    return ctx.world["departments"][f"{ctx.args['institution_id']}_{ctx.args['dept']}"]


def get_department_info(ctx):
    """get_department_info（查一科的資訊，附帶它有哪些服務項目）。
      - 科別這一層藏的是「要先看過初診」這類前置條件，例如大腸鏡要先
        初診、而且檢查必須排在初診的幾天之後
      - 這一科存在但沒公布資訊的話，回一句固定文字。那跟「這一科不存在」
        是兩件事，後者在前置條件就被擋掉了
    """
    dep = _dept(ctx)
    if dep.get("status") == "empty":
        return {"institution_id": dep["institution_id"], "dept": dep["dept"],
                "name": dep.get("name"),
                "information": [{"id": None, "text": render("dept_empty")}],
                "services": []}
    svcs = [{"code": s["code"], "name": s.get("name")}
            for key, s in sorted(ctx.world["services"].items())
            if key[0] == dep["dept_id"]]
    return {"institution_id": dep["institution_id"], "dept": dep["dept"],
            "name": dep.get("name"),
            "information": _blocks(ctx.world, dep.get("block_ids")),
            "services": svcs}


def get_service_info(ctx):
    """get_service_info（查一個服務項目的資訊）。
    項目這一層藏的是準備事項，例如清腸藥須本人到本院藥局領、
    最晚檢查兩天前。
    """
    dep = _dept(ctx)
    svc = ctx.world["services"][(dep["dept_id"], ctx.args["service"])]
    return {"institution_id": dep["institution_id"], "dept": dep["dept"],
            "service": svc["code"], "name": svc.get("name"),
            "information": _blocks(ctx.world, svc.get("block_ids"))}


#  診次

def query_slots(ctx):
    """query_slots（查某個服務項目在一段日期裡還有空位的診次）。
      - 只列還有空位的，不回傳容量數字。容量是醫院的內部資訊
      - 上午／下午一定要回。環境不會替病人過濾掉下午的號
      - 不檢查這個項目適不適合這個病人。拿骨科的手術診次去查也會回傳
        真實的號源
    """
    dep = _dept(ctx)
    lo, hi = ctx.args["date_from"], ctx.args["date_to"]
    out = []
    for sid, s in sorted(ctx.world["slots"].items()):
        if s["dept_id"] != dep["dept_id"] or s["service"] != ctx.args["service"]:
            continue
        if not (lo <= s["date"] <= hi) or s["status"] != "open":
            continue
        if s["booked_count"] >= s["capacity"]:
            continue
        out.append({"slot_id": sid, "date": s["date"],
                    "weekday": weekday(s["date"]), "period": s["period"]})
    return {"slots": out}


#  交通與日期

def _place(ctx, token: str) -> str:
    """_place（把傳進來的字串解析成一個地點代號）。
      - 機構代號與地點代號都接受
      - 病人家沒有機構代號，只能用地點代號，那個代號寫在病歷裡
    """
    if token in ctx.world["places"]:
        return token
    inst = ctx.world["institutions"].get(token)
    if inst is not None and inst.get("place_id") in ctx.world["places"]:
        return inst["place_id"]
    raise ToolError("not_found", f"不認得的地點 {token!r}")


# 機構的地點代號。查交通時間時必須有一端是機構。
def _institution_places(world) -> set[str]:
    return {i["place_id"] for i in world["institutions"].values()
            if i.get("place_id")}


def get_travel_time(ctx):
    """get_travel_time（兩地之間的單程分鐘數）。
      - 回傳的是單程，工具不替 Agent 乘二，也不加總
      - 參數規則：恰好一端是機構（C0 或 H1–H4），另一端不是。不符合
        回 bad_arguments
      - 交通表以機構為軸，所以住處到住處、機構到機構都沒有資料
    """
    a, b = _place(ctx, ctx.args["from"]), _place(ctx, ctx.args["to"])
    inst = _institution_places(ctx.world)
    if (a in inst) == (b in inst):
        raise ToolError(
            "bad_arguments",
            "one end must be an institution and the other must not be")
    mins = ctx.world["travel"].get((a, b))
    if mins is None:
        raise ToolError("not_found", f"沒有 {a} 到 {b} 的交通資料")
    return {"from": a, "to": b, "minutes": mins}


def date_calc(ctx):
    """date_calc（日期加減，或算兩個日期相差幾天）。一律附上星期幾。
      - 一定要回星期幾：醫院的規定全部寫成週二、四上午這種形式
    """
    d = ctx.args["date"]
    if ctx.args.get("add_days") is not None:
        out = shift(d, int(ctx.args["add_days"]))
        return {"date": out, "weekday": weekday(out), "days": None}
    other = ctx.args["date_to"]
    return {"date": d, "weekday": weekday(d), "days": diff(other, d)}
