"""指引檢索的工具：Agent 查臨床依據的唯一管道。
Agent 給一段查詢詞，拿回幾份條件式指引的全文，
自己比對病人的條件、自己挑出哪一份適用、自己把數字抽出來。

這支工具刻意做得很笨：
  - 不報錯：查詢詞再離題也照樣回固定筆數，不回空清單
  - 不提醒：撈回來的幾份裡本來就有不適用的，它不標註哪一份適用
  - 不推理：不做查詢改寫、不做同義詞擴展、不根據病歷補條件
  - 不回分數：排名順序已經是一種引導，不再加一個相關性數字

回傳的欄位只有五個：guideline_id / decision / title / when / action。
assumption / source / reviewed_by 絕對不回傳，過濾用的是白名單
（loaders/guidelines.py 的 AGENT_FIELDS）。

全域層級：不管站在哪個節點都能查。
"""
from __future__ import annotations

from ...loaders.guidelines import for_agent
from ..envelope import ToolError
from ..retrieval import DEFAULT_TOP_K, MAX_TOP_K, Index

# 索引只建一次。指引庫在一次執行裡不會變，而且建索引是純函數。
_INDEX: dict[int, tuple[Index, dict]] = {}


def _index(guidelines: dict) -> tuple[Index, dict]:
    """_index（拿到這一題指引庫的索引）：第一次會建，之後直接重用。
    一併回傳 Agent 讀得到的那份字典，因為索引就是用它建的。
    """
    key = id(guidelines)
    if key not in _INDEX:
        visible = {gid: for_agent(g) for gid, g in guidelines.items()}
        _INDEX[key] = (Index(visible), visible)
    return _INDEX[key]


def search_guidelines(ctx):
    """search_guidelines（檢索指引），最相關的在前。
    永遠回滿筆數（指引庫夠大的話），即使分數全是 0。
    """
    k = ctx.args.get("top_k") or DEFAULT_TOP_K
    if not isinstance(k, int) or k < 1 or k > MAX_TOP_K:
        raise ToolError("bad_arguments", f"top_k 必須是 1 到 {MAX_TOP_K} 之間")
    g = ctx.world["guidelines"]
    index, visible = _index(g)
    hits = index.search(str(ctx.args.get("query") or ""), k)
    return {"results": [visible[gid] for gid in hits]}
