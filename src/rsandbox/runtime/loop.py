"""主迴圈：一個案例從第一天跑到最後一天。
    當 今天 還沒超過 結束日：
        觸發今天的事件 -> 組出觀察 -> 喚醒 Agent 跑一個回合 -> 推進時鐘

時鐘只在 Agent 呼叫 end_turn（結束回合）時前進，每次前進的天數寫在
任務檔的 clock 裡。單一 Agent、沒有事件佇列、模擬代理人不會主動行動。

主迴圈不做任何判斷。它不檢查 Agent 做得對不對、不提醒它漏了什麼、
也不因為狀態不一致而插話。它只做三件事：
  - 報今天幾號
  - 轉述發生了什麼事
  - 把時間收回來

Agent 這一端的介面只有一個方法：

    agent.activate(session, observation) -> None
    （activate ＝ 喚醒。observation ＝ 這回合給它看的觀察文字）

    呼叫一次 ＝ 一個回合。Agent 自己決定這回合要做幾件事，
    做完呼叫 end_turn。照腳本跑的 Agent 與真模型共用這個介面。
"""
from __future__ import annotations

from . import initial
from .messages import EVENT_TEMPLATE, render

# MAX_TURNS（回合數上限）。保險絲而已：正常案例最多用到 5 個回合。
MAX_TURNS = 40

# IDLE_TURN_LIMIT（連續幾個回合沒有任何寫入就停）。
#   讀資料也算在做事（見 _acted_this_turn），所以一個一直重複查案件狀態
#   的模型會一路跑到結束日。這道保險絲擋住那種情況，正常流程踩不到。
IDLE_TURN_LIMIT = 3

# WRITE_TOOLS（真正會改變世界的工具）。
#   - 只用在上面那道保險絲，不是用來判斷「這回合有沒有做事」
#   - record_assessment 不在這份清單裡：它寫的是 Agent 自己的記事本，
#     不是世界
WRITE_TOOLS = ("book_appointment", "reschedule_appointment", "issue_hold_order",
               "order_preop_lab", "send_prep_instructions",
               "update_patient_contact")


def events_on(task: dict, day: str) -> list[dict]:
    """events_on（今天要觸發哪些事件），照任務檔寫的順序，所以可重現。"""
    return [ev for ev in task.get("event_script") or [] if ev["date"] == day]


def future_events(task: dict, day: str) -> list[dict]:
    """future_events（今天之後還有沒有排定的事件）。"""
    return [ev for ev in task.get("event_script") or [] if ev["date"] > day]


def build_observation(session, task: dict, events: list[dict], turn: int) -> str:
    """build_observation（組出這一回合 Agent 看到的那段文字）。
    只有三種內容：今天幾號、案件簡報（只在第一回合）、事件原文。
      - 事件內容逐字轉述，不摘要、不加註、不解讀
      - 所有模板都放在 messages.py
    """
    parts = [render("obs_today", date=session.today)]
    if turn == 0:
        parts.append(render("obs_opened", patient_id=task["patient_id"]))
        # 觀察只給病人編號，不附病歷全文。Agent 要自己呼叫
        # get_patient_record 把它叫出來。
    for ev in events:
        key = EVENT_TEMPLATE.get(ev["type"])
        if key is None:
            raise ValueError(f"不認得的事件型別 {ev['type']!r}")
        parts.append(render(key, content=ev["content"]))
    return "\n".join(p for p in parts if p)


def _acted_this_turn(session, before: int) -> bool:
    """_acted_this_turn（這個回合有沒有做任何事）。
    判準：
      - 任何成功的工具呼叫都算，不是只有寫入才算
      - end_turn 不算，它的意思是「我這輪講完了」
      - 讀病歷、查指引都是在做事。只有完全空的回合才代表它沒事做了
    """
    return any(r["ok"] and r["tool"] != "end_turn"
               for r in session.state["tool_call_log"][before:])


def _wrote_this_turn(session, before: int) -> bool:
    """_wrote_this_turn（這個回合有沒有真的推進世界）。
    只用在「連續幾個回合沒推進就停」那道保險絲上。
    """
    return any(r["ok"] and r["tool"] in WRITE_TOOLS
               for r in session.state["tool_call_log"][before:])



# PATIENT_ORIGINATED：哪些事件是病人自己做的。
# 只有這些會追加到病人的個人資料裡。醫院取消時段、環境發的提醒，
# 病人都不知道。
PATIENT_ORIGINATED = ("patient_call",)


def share_with_patient(session, events: list[dict]) -> None:
    """share_with_patient（把病人自己做的事追加到他的個人資料）。
      - 轉診 Agent 的觀察與病人的個人資料從同一個事件宣告產生，
        所以病人 Agent 不另外存一份記憶
      - 事件的 content 本來就是病人口吻寫的，所以原樣追加，不改寫
    """
    log = session.patient_doc.setdefault("log", [])
    for ev in events:
        if ev["type"] not in PATIENT_ORIGINATED:
            continue
        log.append({"date": ev["date"],
                    "text": "You telephoned the clinic and said: "
                            + str(ev["content"]).strip()})

def run_task(data: dict, task: dict, agent, base_snapshot: dict | None = None,
             max_turns: int = MAX_TURNS, patient_backend=None) -> dict:
    """run_task（跑完一個案例），回傳跑完的 Session。
    結束條件（任一成立就停）：
          - 時鐘走過了結束日
          - Agent 沒事做了（這回合一次成功的呼叫都沒有，end_turn 不算）
            而且未來已經沒有排定的事件
          - 回合數撞到保險絲
      - 第②條的用意：如果只看結束日，一個案例會從做完事的那天一路空轉
        好幾週，而每一個空回合都給 Agent 一次「再去亂改點什麼」的機會，
        那會稀釋「不該動的時候不要動」這種對照組。
        Agent 一整個回合什麼都沒碰，就該結束。
      - 但「沒事做」的判準必須是任何工具呼叫，不能只算寫入，
        理由見 _acted_this_turn。
    """
    session = initial.build(data, task, base_snapshot,
                            patient_backend=patient_backend)
    clock = task["clock"]
    turn = 0
    idle = 0
    session.state.setdefault("observations", [])

    while session.today <= clock["end_date"] and turn < max_turns:
        events = events_on(task, session.today)
        # 同一份事件宣告供兩邊：轉診 Agent 的觀察，與病人的個人資料。
        share_with_patient(session, events)
        obs = build_observation(session, task, events, turn)
        session.state["observations"].append(
            {"turn": turn, "at": session.today, "text": obs,
             "events": [ev["type"] for ev in events]})

        before = len(session.state["tool_call_log"])
        agent.activate(session, obs)
        turn += 1

        # 「沒事做了」由誰來說：
        #   - 照腳本跑的 Agent 知道腳本走完了沒有，直接問它
        #   - 真模型沒有這個概念，用上面那兩個判準去推
        # 有些走法必須中間空一個回合（做完事 → 空一天 → 隔天才寄新單），
        # 所以不能一看到空回合就結束。
        if hasattr(agent, "finished"):
            done = agent.finished()
        else:
            done = not _acted_this_turn(session, before)
            idle = 0 if _wrote_this_turn(session, before) else idle + 1
            if idle >= IDLE_TURN_LIMIT:
                done = True        # 連續幾個回合只讀不寫 = 原地打轉
        if done and not future_events(task, session.today):
            break
        session.advance(int(clock.get("days_per_turn", 1)))
        # 每一天都從診所開始：一天之內走到哪都算，跨日回到起點。
        session.state["position"]["node"] = task.get("start_node") or "C0"

    session.state["turns"] = turn
    return session
