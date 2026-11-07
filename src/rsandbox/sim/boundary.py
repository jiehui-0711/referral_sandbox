"""模擬代理人的越線紀錄。
病人每說一句話都要經過 guard()。
違規不丟例外，而是記進狀態的 boundary_violations，trace 與 show-run
會印出來。那一輪的結果就不該採用。

病人由一個 LLM 扮演，該說什麼、不該說什麼全部寫在
data/prompts/patient_agent.txt 裡，沒有程式層的規則判斷。
"""
from __future__ import annotations


def record(session, agent_id: str, errs: list[str]) -> None:
    """record（把違規記進狀態）。"""
    if not errs:
        return
    session.state.setdefault("boundary_violations", [])
    for e in errs:
        session.state["boundary_violations"].append(
            {"at": session.today, "agent_id": agent_id, "violation": e})


def guard(session, reply: str) -> None:
    """guard（病人每說一句話都會經過這裡），目前不做任何檢查。
    留著這個入口，之後要加語意層的偵測（用另一個模型判斷病人有沒有
    主動提示）就接在這裡，呼叫端一行都不用改。
    """
    return None
