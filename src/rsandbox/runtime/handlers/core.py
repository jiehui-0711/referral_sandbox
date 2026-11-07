"""控制類的工具。"""
from __future__ import annotations


def end_turn(ctx):
    """end_turn（結束這一回合，把時間交還給世界）。
    這支 handler 本身什麼都不做，真正推進時鐘的是主迴圈。
    note 是 Agent 自己寫的備註，會進呼叫紀錄。
    """
    return {"ended": True, "note": ctx.args.get("note")}
