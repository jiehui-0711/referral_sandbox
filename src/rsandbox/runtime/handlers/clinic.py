"""轉出端診所能做的事（Agent 自己的診所，節點 C0_FM 與 C0_ADM）。"""
from __future__ import annotations

from ...sim import boundary
from ...sim.patient import answer as patient_answer
from .. import casework
from ..messages import render

# AUTHORISER：停藥醫囑上的簽章是誰蓋的。
# 醫師沒有行為、不說話，所以它不是一個模擬代理人，只是這個固定的值。
AUTHORISER = "doctor.D001"


def _ask(ctx, question: str) -> str:
    """_ask（問病人一句話，拿回他的回覆）。
      - 病人是一個 LLM，讀的是這一組情境的個人視角資料，不是病歷
      - 行為規則寫在 data/prompts/patient_agent.txt，不寫成程式判斷
      - 它看得到 contact_log 裡之前的問答，所以答得出追問
    """
    reply = patient_answer(
        ctx.patient_doc, question,
        today=ctx.today,
        instructions=ctx.patient_instructions,
        contact_log=ctx.state["contact_log"],
        backend=ctx.patient_backend)
    boundary.guard(ctx.session, reply)
    return reply


def ask_patient(ctx):
    """ask_patient（問病人一個問題）。
    問得夠不夠精準決定問不問得出來，病人只回答被問到的。
    """
    q = ctx.args["question"]
    reply = _ask(ctx, q)
    for direction, content in (("outbound", q), ("inbound", reply)):
        cid = ctx.new_id("contact_log")
        ctx.state["contact_log"][cid] = {
            "log_id": cid, "direction": direction,
            "content": content, "created_at": ctx.today,
        }
    return {"answer": reply}


def get_case_status(ctx):
    """get_case_status（看這個案子目前的全貌）。
    回傳目前的轉診單、預約、療程、停藥醫囑、抽血單、已寄出的說明單、
    已上報的問題。

    Agent 從這裡同時看得到預約日與末次服藥日這兩個值，
    但沒有任何欄位告訴它這兩者有關。

    開頭帶這一案的病人是誰，所以 Agent 隨時查得到，不必靠記憶。
    """
    def rows(table):
        """把一張表攤成清單，依代號排序（所以每次看到的順序都一樣）。"""
        return [dict(r) for _, r in sorted(ctx.state[table].items())]
    record = ctx.records[ctx.patient_id]
    return {
        "patient_id": ctx.patient_id,
        "patient_name": record.get("name"),
        "referrals": rows("referrals"),
        "appointments": rows("appointments"),
        "series": rows("series"),
        "hold_orders": rows("hold_orders"),
        "lab_orders": rows("lab_orders"),
        "prep_instructions": rows("prep_instructions"),
        "escalations": rows("escalations"),
    }


def record_assessment(ctx):
    """record_assessment（記下自己算出了什麼、依據是什麼）。
      - 寫進紀錄的是：算出什麼結論，以及引用了病歷的哪些欄位
      - 「依據哪一條規則」那一欄是選填的
    """
    aid = ctx.new_id("assessments")
    ctx.state["assessments"][aid] = {
        "assessment_id": aid,
        "key": ctx.args["key"],
        "value": ctx.args["value"],
        "evidence": list(ctx.args["evidence"]),
        "rule_id": ctx.args.get("rule_id"),
        "created_at": ctx.today,
    }
    return {"assessment_id": aid}


def issue_hold_order(ctx):
    """issue_hold_order（開立停藥醫囑）。
    末次服藥日（last_dose_date）完全不被驗證：
      - 不驗它跟預約日的關係
      - 不驗它跟停藥天數的關係
      - 不驗它是不是已經過去了
    傳一個跟預約日完全對不起來的值照樣成功。

    不填「要取代哪一張」就會留下兩張生效中的醫囑，環境不阻止。
    """
    prev = ctx.args.get("supersedes")
    if prev is not None:
        ctx.state["hold_orders"][prev]["status"] = "superseded"
    oid = ctx.new_id("hold_orders")
    order = {
        "order_id": oid,
        "drug": ctx.args["drug"],
        "hold_days": ctx.args["hold_days"],
        "last_dose_date": ctx.args["last_dose_date"],
        "substitute": ctx.args["substitute"],
        "status": "active",
        "supersedes": prev,
        "issued_at": ctx.today,
    }
    ctx.state["hold_orders"][oid] = order
    # 醫師核可：無條件通過，不檢查內容、不提醒任何事。即使這張醫囑的
    # 末次服藥日在三天前（病人根本做不到）也照樣簽。
    # 要加條件就在這裡加：目前沒有任何條件。
    return {"order_id": oid, "status": "active",
            "authorised_by": AUTHORISER,
            "authorisation": render("doctor_approved")}


def send_prep_instructions(ctx):
    """send_prep_instructions（寄準備說明單給病人）。最重要的一支工具。
    快照：寄出的那一刻，把目前所有已訂的預約、所有生效中的停藥醫囑與
    抽血單複製一份存進這筆紀錄。之後預約改期、或醫囑被取代，這份快照
    都不會變。規則跟轉診單的快照一致。

    寄什麼由狀態決定，不由參數決定：
      - 不收「要引用哪幾張單子」這種參數
      - 寄的是呼叫這一刻生效中的那些單子，依藥名與檢驗項目分項列出，
        互不覆寫
      - 已經被取代或撤銷的單子不會出現在快照裡
      - Agent 如果在改醫囑之前就寄，快照存下來的就是舊的

    這支工具只負責把快照留下來。病人收到之後實際怎麼做，環境不模擬。
    """
    ap = casework.primary_appointment(ctx.state)
    holds = {drug: {"order_id": rows[0]["order_id"],
                    "hold_days": rows[0]["hold_days"],
                    "last_dose_date": rows[0]["last_dose_date"],
                    "substitute": rows[0]["substitute"]}
             for drug, rows in casework.active_holds_by_drug(ctx.state).items()}
    labs = {test: {"order_id": rows[0]["lab_order_id"],
                   "draw_date": rows[0]["draw_date"],
                   # 抽血地點是第三個衍生值：換醫院時它也要跟著改，
                   # 因為有些醫院不收外院的報告。
                   "site": rows[0].get("site")}
            for test, rows in casework.active_labs_by_test(ctx.state).items()}

    iid = ctx.new_id("prep_instructions")
    ctx.state["prep_instructions"][iid] = {
        "instruction_id": iid,
        "delivered_at": ctx.today,
        "snapshot": {
            # 主處置的預約日。
            "procedure_date": ap["date"] if ap and ap["status"] in casework.LIVE
                              else None,
            "appointments": [{"appointment_id": a["appointment_id"],
                              "slot_type": a["slot_type"], "date": a["date"]}
                             for a in casework.booked_appointments(ctx.state)],
            "holds": holds,
            "lab_orders": labs,
            # 只停一種藥的時候，這一欄直接給出那個日期；
            # 多種藥的時候留空，要去上面的 holds 逐藥讀。
            "last_dose_date": (sorted(holds.values(),
                                      key=lambda h: h["order_id"])[0]["last_dose_date"]
                               if len(holds) == 1 else None),
        },
    }
    return {"instruction_id": iid, "delivered_at": ctx.today,
            "withhold": sorted(holds), "blood_tests": sorted(labs)}


def update_patient_contact(ctx):
    """update_patient_contact（更新病人的聯絡電話）。
    「病人換電話了」這個擾動該做的事就只有改電話，不該連帶重算任何
    日期。環境不檢查這件事。
    """
    # 目前一個案子只有一個病人，直接取任務指定的那一位。
    pid = ctx.session.task["patient_id"]
    ctx.records[pid]["phone"] = ctx.args["phone"]
    return {"phone": ctx.args["phone"]}


def consult_patient(ctx):
    """consult_patient（把整理好的方案寄給病人，並問他一個問題）。
      - 整理內容裡要附上引用了哪幾個資訊段落的編號。紀錄下來的是那些
        編號，不是 Agent 自己算出的趟數或分鐘數
      - 病人怎麼回答跟 ask_patient 完全一樣，沒有特例
    """
    q = ctx.args["question"]
    reply = _ask(ctx, q)
    sid = ctx.new_id("summaries")
    ctx.state["summaries"][sid] = {
        "summary_id": sid,
        "summary": [dict(o) for o in ctx.args["summary"]],
        "question": q,
        "reply": reply,
        "created_at": ctx.today,
    }
    for direction, content in (("outbound", q), ("inbound", reply)):
        cid = ctx.new_id("contact_log")
        ctx.state["contact_log"][cid] = {
            "log_id": cid, "direction": direction,
            "content": content, "created_at": ctx.today,
        }
    return {"summary_id": sid, "reply": reply}


def cancel_order(ctx):
    """cancel_order（撤銷一張停藥醫囑或抽血單）。
    撤銷不等於取代。處置被取消的時候（例如血糖太高、手術延後），
    停藥醫囑要撤回，而不是換一張新的。
    """
    oid = ctx.args["order_id"]
    row = ctx.state["hold_orders"].get(oid) or ctx.state["lab_orders"][oid]
    row["status"] = "cancelled"
    row["cancelled_at"] = ctx.today
    return {"order_id": oid, "status": "cancelled"}


def escalate_to_physician(ctx):
    """escalate_to_physician（把一個臨床問題上報給醫師）。
      - 同一個決定恰好有一份指引適用，零份也可以，那代表正確答案就是
        上報醫師。指引庫確實有涵蓋不到的病人：年齡落在篩檢範圍之外、
        抗凝血藥的用藥原因不屬於指引涵蓋的任何一種
      - 上報不會把 Agent 卡住：回傳的訊息明說醫師不會回覆，所有任務
        都是同一句
    """
    eid = ctx.new_id("escalations")
    ctx.state["escalations"][eid] = {
        "escalation_id": eid,
        "question": ctx.args["question"],
        "evidence": list(ctx.args["evidence"]),
        "created_at": ctx.today,
    }
    return {"escalation_id": eid, "message": render("escalated")}
