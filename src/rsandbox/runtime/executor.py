"""關卡：一次工具呼叫要通過的檢查。
順序固定，而且順序本身有意義：

    1. 這支工具存在嗎        -> bad_arguments
    2. 有權限用嗎            -> permission_denied / not_available_here
       （含「站在這裡能不能用」，目前角色只有一個，角色那半永遠通過）
    3. 參數格式對嗎          -> bad_arguments
    4. 行政前置條件成立嗎    -> 工具檔自己宣告的失敗理由
    5. 執行 handler          -> 成功

第 3 關是格式檢查，第 4 關是行政檢查，沒有第六關。工具不會因為
「臨床上不適當」而擋下呼叫：
  - 傳一個跟預約日對不起來的停藥日期：放行
  - 拿門診的號去約要做內視鏡的處置：放行

前置條件的宣告與實作分開：
  - 宣告在工具的 YAML 檔裡，讀那個檔就看得到這支工具有哪幾道關卡
  - 實作在下面的 PREDICATES
  - 兩邊一一對應。YAML 寫了一道 PREDICATES 沒有的關卡，執行到那裡會丟錯
"""
from __future__ import annotations

import re

from .envelope import ToolError

_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


#  參數格式

def check_value(value, spec: dict, label: str) -> list[str]:
    """check_value（檢查一個值的型別對不對），回傳問題描述的清單。
    會往裡面鑽：物件與清單的內部結構也一起檢查。
    錯誤訊息會把允許的欄位列出來，跟工具定義裡寫的一致。
    """
    t = spec.get("type", "string")
    if value is None:
        return [] if spec.get("nullable") else [f"{label}: must not be null"]

    if t == "string" and not isinstance(value, str):
        return [f"{label}: expected a string"]
    if t == "int" and (not isinstance(value, int) or isinstance(value, bool)):
        return [f"{label}: expected an integer"]
    if t == "bool" and not isinstance(value, bool):
        return [f"{label}: expected true or false"]
    if t == "date" and not (isinstance(value, str) and _DATE.match(value)):
        return [f"{label}: expected a date as YYYY-MM-DD"]

    if t == "object":
        if not isinstance(value, dict):
            return [f"{label}: expected an object"]
        props = spec.get("properties")
        if not props:
            return []
        problems = []
        for k in value:
            if k not in props:
                problems.append(f"{label}.{k}: unknown field; allowed fields are {sorted(props)}")
        for k, sub in props.items():
            if k not in value:
                if sub.get("required"):
                    problems.append(f"{label}.{k}: missing required field")
                continue
            problems += check_value(value[k], sub, f"{label}.{k}")
        return problems

    if t == "list":
        if not isinstance(value, list):
            return [f"{label}: expected a list"]
        item = spec.get("items")
        if not item:
            return []
        problems = []
        for i, v in enumerate(value):
            problems += check_value(v, item, f"{label}[{i}]")
        return problems

    if spec.get("enum") and value not in spec["enum"]:
        # 這幾句會送到 Agent 那一側：不回傳傳進來的值，不列合法值。
        return [f"{label}: not an allowed value"]
    return []


def validate_args(tool: dict, args: dict) -> None:
    """validate_args（第 3 關：檢查參數）。
      - 值不在允許清單裡算格式錯誤，不是臨床判斷
      - 問題清單兩個用途：detail 給開發者，fields 給 Agent
    """
    params = tool.get("params") or {}
    problems = []
    for name in args:
        if name not in params:
            problems.append(f"unknown parameter {name!r}")
    for name, spec in params.items():
        if name not in args:
            if spec.get("required"):
                problems.append(f"missing required parameter {name!r}")
            continue
        problems += check_value(args[name], spec, name)
    if problems:
        raise ToolError("bad_arguments", "; ".join(problems), fields=problems)


#  行政前置條件

def _slot(ctx, key: str):
    """_slot（從參數裡取出的診次代號，去世界裡把那個診次撈出來）。"""
    return ctx.world["slots"].get(ctx.args.get(key))


# 位置關卡。它落在第 2 關（權限），問的是「你現在站的地方能不能做這件事」。

def _here(ctx) -> str | None:
    """_here（Agent 現在站在哪個節點）。"""
    return (ctx.state.get("position") or {}).get("node")


def _node(ctx, nid):
    """_node（照節點代號把那個節點撈出來）。"""
    return ctx.session.nodes.get(nid or "")


def gate_allows(ctx) -> bool:
    """gate_allows（這支工具能不能在現在站的地方用）。
    由工具所在的資料夾（data/tools/<層級>/）決定，不逐一列節點。
    """
    here = _node(ctx, _here(ctx))
    if here is None:
        return False
    return ctx.tool["tool_id"] in (here.get("tools") or [])


def _dept_id(ctx) -> str | None:
    """_dept_id（參數指的是哪一科）。
    科別的完整代號是「機構代號 ＋ 底線 ＋ 科別縮寫」，例如 H1_GI。
    沒有傳科別的工具回空值。
    """
    iid, dept = ctx.args.get("institution_id"), ctx.args.get("dept")
    if iid and dept:
        return f"{iid}_{dept}"
    return None


def _referral_ok(ctx) -> bool:
    """_referral_ok（這家醫院規定訂號前要有轉診單嗎？有的話找得到嗎）。
    純行政。它不檢查轉診是不是轉去對的科：轉到大腸直腸外科、再訂那一科的
    門診號，一路都會成功，診次型別不對這件事只會留在最終資料裡。
    """
    sid = ctx.args.get("slot_id")
    if sid:
        slot = ctx.world["slots"].get(sid) or {}
        iid = slot.get("institution_id")
    else:
        iid = ctx.args.get("institution_id")
    inst = ctx.world["institutions"].get(iid) or {}
    if not inst.get("booking_requires_referral"):
        return True
    return any(r["status"] == "active" and r["institution_id"] == iid
               for r in ctx.state["referrals"].values())


def _same_service(ctx) -> bool:
    """_same_service（新舊診次是不是同一個服務項目）。
    改期只能在同一個項目之間移動。換醫院要走四步：
    取消 → 撤回轉診 → 送新轉診 → 重新訂號。
    """
    ap = ctx.state["appointments"].get(ctx.args.get("appointment_id")) or {}
    new = ctx.world["slots"].get(ctx.args.get("new_slot_id")) or {}
    return (ap.get("institution_id"), ap.get("dept"), ap.get("service")) == \
           (new.get("institution_id"), new.get("dept"), new.get("service"))


def _series_fits(ctx) -> bool:
    """_series_fits（一整組療程的號排得下嗎）。
    排不下就一筆都不寫：整組訂，或整組不訂。
    """
    from .handlers.hospital import series_slots
    try:
        return len(series_slots(ctx)) >= int(ctx.args.get("count") or 0)
    except Exception:
        return False


def _cancel_target(ctx):
    """_cancel_target（要取消的是哪一筆）：一整組療程，或單一筆預約。"""
    if ctx.args.get("series_id"):
        return ctx.state["series"].get(ctx.args["series_id"])
    return ctx.state["appointments"].get(ctx.args.get("appointment_id"))


def _cancellable(ctx) -> bool:
    """_cancellable（這張單子還撤得掉嗎）。
    已經抽過的血不能撤銷。純行政：那件事實際上已經發生了。
    """
    oid = ctx.args.get("order_id")
    row = ctx.state["hold_orders"].get(oid) or ctx.state["lab_orders"].get(oid) or {}
    return row.get("status") not in ("drawn", "resulted", "cancelled")


# PREDICATES（判斷式表）：工具檔裡寫的那句宣告字串 -> 真正的判斷函式。
#
# 每一條都只看行政事實，一條都不碰臨床正確性。
PREDICATES: dict[str, object] = {
    #  位置與節點
    "args.node exists":
        lambda ctx: ctx.args.get("node") in ctx.session.nodes,
    "args.node is adjacent":
        lambda ctx: ctx.args.get("node") in
                    ((_node(ctx, _here(ctx)) or {}).get("neighbours") or []),
    "args.institution_id in institutions":
        lambda ctx: ctx.args.get("institution_id") in ctx.world["institutions"],
    "args.department exists":
        lambda ctx: _dept_id(ctx) in ctx.world["departments"],
    "args.service exists":
        lambda ctx: (_dept_id(ctx), ctx.args.get("service")) in ctx.world["services"],
    "args.place pair exists":
        lambda ctx: True,          # 這條交給 handler 判（代號要先解析才知道）

    #  轉診
    "args.institution is referable":
        lambda ctx: bool((ctx.world["institutions"].get(ctx.args.get("institution_id"))
                          or {}).get("referable")),
    "args.referral_id in referrals":
        lambda ctx: ctx.args.get("referral_id") in ctx.state["referrals"],
    "referral_is_active":
        lambda ctx: (ctx.state["referrals"].get(ctx.args.get("referral_id"))
                     or {}).get("status") == "active",
    "referral_exists_if_required": lambda ctx: _referral_ok(ctx),

    #  預約
    "same_service": lambda ctx: _same_service(ctx),
    "series_fits": lambda ctx: _series_fits(ctx),
    "args.appointment_id xor args.series_id":
        lambda ctx: bool(ctx.args.get("appointment_id")) != bool(ctx.args.get("series_id")),
    "cancel target exists": lambda ctx: _cancel_target(ctx) is not None,

    #  醫囑
    "args.site in institutions":
        lambda ctx: ctx.args.get("site") in ctx.world["institutions"],
    "args.order_id in orders":
        lambda ctx: (ctx.args.get("order_id") in ctx.state["hold_orders"]
                     or ctx.args.get("order_id") in ctx.state["lab_orders"]),
    "order_is_cancellable": lambda ctx: _cancellable(ctx),

    #  其他
    "args.add_days xor args.date_to":
        lambda ctx: (ctx.args.get("add_days") is None) != (ctx.args.get("date_to") is None),
    "args.summary is well formed":
        lambda ctx: isinstance(ctx.args.get("summary"), list) and bool(ctx.args["summary"]),

    # 調閱病歷

    # 訂號
    "args.slot_id in slots":
        lambda ctx: ctx.args.get("slot_id") in ctx.world["slots"],
    "slot.status == open":
        lambda ctx: (_slot(ctx, "slot_id") or {}).get("status") == "open",
    "slot.date > today":
        lambda ctx: (_slot(ctx, "slot_id") or {}).get("date", "") > ctx.today,
    "slot.booked_count < slot.capacity":
        lambda ctx: (_slot(ctx, "slot_id") or {}).get("booked_count", 1)
                    < (_slot(ctx, "slot_id") or {}).get("capacity", 0),

    # 改期
    # 改期一定要明寫是哪一筆預約，不讓 handler 自己去抓「第一筆已訂的」。
    "args.appointment_id in appointments":
        lambda ctx: ctx.args.get("appointment_id") in ctx.state["appointments"],
    "appointment.status == booked":
        lambda ctx: (ctx.state["appointments"].get(ctx.args.get("appointment_id"))
                     or {}).get("status") == "booked",
    "args.new_slot_id in slots":
        lambda ctx: ctx.args.get("new_slot_id") in ctx.world["slots"],
    "new_slot.status == open":
        lambda ctx: (_slot(ctx, "new_slot_id") or {}).get("status") == "open",
    "new_slot.date > today":
        lambda ctx: (_slot(ctx, "new_slot_id") or {}).get("date", "") > ctx.today,
    "new_slot.booked_count < new_slot.capacity":
        lambda ctx: (_slot(ctx, "new_slot_id") or {}).get("booked_count", 1)
                    < (_slot(ctx, "new_slot_id") or {}).get("capacity", 0),

    # 開立停藥醫囑
    "args.supersedes is null or args.supersedes in hold_orders":
        lambda ctx: ctx.args.get("supersedes") is None
                    or ctx.args["supersedes"] in ctx.state["hold_orders"],
    "superseded_target_is_active":
        lambda ctx: ctx.args.get("supersedes") is None
                    or ctx.state["hold_orders"][ctx.args["supersedes"]]["status"] == "active",

    # 開立術前抽血單（取代舊單的規則跟停藥醫囑一致）
    "args.supersedes is null or args.supersedes in lab_orders":
        lambda ctx: ctx.args.get("supersedes") is None
                    or ctx.args["supersedes"] in ctx.state["lab_orders"],
    "superseded_lab_is_active":
        lambda ctx: ctx.args.get("supersedes") is None
                    or ctx.state["lab_orders"][ctx.args["supersedes"]]["status"] == "active",

    # 角色權限。由載入器自動補到每一支工具上，目前只有一個角色，永遠通過。
    "caller.role in tool.permissions":
        lambda ctx: ctx.role in (ctx.tool.get("permissions") or []),
}


def check_gate(ctx) -> bool:
    """check_gate（第 2 關的位置部分：站在這裡能不能用這支工具）。
      - 跟角色權限同一關。問的是「你現在在哪裡」，純粹行政
      - 擋下來一律回 not_available_here，訊息固定，不提示該去哪裡
    """
    return gate_allows(ctx)


def check_preconditions(ctx) -> None:
    """check_preconditions（第 2 與第 4 關）：照工具檔列的順序一條條檢查。
    清單順序就是檢查順序，第一條不成立就停。工具檔的慣例是先寫
    「這個代號存在嗎」，再寫「那筆東西的狀態對嗎」。
    """
    for pre in ctx.tool.get("precondition") or []:
        rule = pre["rule"]
        fn = PREDICATES.get(rule)
        if fn is None:
            # 載入器已經擋掉了；這裡只是最後一道保險。
            raise ToolError("bad_arguments", f"前置條件 {rule!r} 沒有實作")
        if not fn(ctx):
            raise ToolError(pre["on_fail"], rule)
