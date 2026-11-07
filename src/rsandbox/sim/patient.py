"""病人：由一個 LLM 扮演。
  - 它知道的事全部在情境資料夾的 patient_doc.yaml 裡，一份第一人稱的
    散文。病歷它讀不到，病歷是給轉診 Agent 查的
  - 只回答被問到的、不主動多說、沒寫到的就說不知道，這些規則寫在
    data/prompts/patient_agent.txt，不寫成程式判斷
  - 它看得到這一案之前的問答（從 contact_log 來），所以答得出追問。
    但只看得到日期 ≤ 今天的事件
  - 沒有設定病人模型的時候回 OFFLINE_REPLY，讓腳本照樣跑得完
"""
from __future__ import annotations

OFFLINE_REPLY = "(no patient model configured)"

# 送給病人模型的最大歷史則數（一問一答算兩則）。太長會拖慢，也沒必要。
MAX_HISTORY = 20


def visible_log(doc: dict, today: str) -> list[dict]:
    """visible_log（病人今天為止知道發生過哪些事）。
    日期晚於今天的條目不給它看。
    """
    return [e for e in (doc.get("log") or []) if e["date"] <= today]


def knowledge_text(doc: dict, today: str) -> str:
    """knowledge_text（把個人資料攤成一段提示詞）。
    節名會印出來（What you know 底下一行一節），因為提示詞要指名
    What matters to me 那一節。
    """
    parts = ["## Who you are", (doc.get("persona") or "").strip()]
    parts.append("\n## What you know")
    for name, text in (doc.get("knows") or {}).items():
        label = name.replace("_", " ").capitalize()
        parts.append(f"\n### {label}\n{str(text).strip()}")
    seen = visible_log(doc, today)
    if seen:
        parts.append("\n## What has happened")
        for e in seen:
            parts.append(f"\n### On {e['date']}\n{str(e['text']).strip()}")
    return "\n".join(parts)


def system_prompt(doc: dict, today: str, instructions: str) -> str:
    """system_prompt（病人模型收到的第一段文字）＝ 行為規則 ＋ 個人資料。"""
    return instructions.rstrip() + "\n\n" + knowledge_text(doc, today) + "\n"


def history_messages(contact_log: dict) -> list[dict]:
    """history_messages（這一案之前的問答，翻成對話歷史）。
    contact_log 裡 outbound 是診所問的、inbound 是病人答的。
    照編號排序，所以順序就是實際發生的順序。
    """
    rows = sorted(contact_log.values(), key=lambda r: r["log_id"])
    msgs = [{"role": "user" if r["direction"] == "outbound" else "assistant",
             "content": r["content"]} for r in rows]
    return msgs[-MAX_HISTORY:]


def answer(doc: dict, question: str, *, today: str, instructions: str,
           contact_log: dict | None = None, backend=None) -> str:
    """answer（病人對一個問題的回答）。
    沒有 backend 就回 OFFLINE_REPLY，不走任何規則判斷。
    """
    if backend is None:
        return OFFLINE_REPLY
    messages = [{"role": "system",
                 "content": system_prompt(doc, today, instructions)}]
    messages += history_messages(contact_log or {})
    messages.append({"role": "user", "content": question})
    reply = backend.chat(messages, [])
    return (reply.get("content") or "").strip() or OFFLINE_REPLY
