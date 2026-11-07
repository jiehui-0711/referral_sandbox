"""組出被測 Agent 的系統提示詞。
系統提示詞 ＝ data/prompts/<角色>.txt 的全文 ＋ 工具清單。
所有任務共用同一份，不隨情境或變體改變。

唯一會被換掉的是講「一則訊息給幾個工具呼叫」那一行，見 CALL_RULE。
"""
from __future__ import annotations

from pathlib import Path

from ..paths import DATA

# ROLES（被測角色）。只有這些角色的提示詞裡有呼叫規則那一行，
# 其餘角色不換也不檢查。
ROLES = ("referral_agent",)

# CALL_RULE（提示詞裡講一則訊息能給幾個呼叫的那一行），一個模式一句。
#   - 鍵是 react.py 的 TOOL_CALL_MODES
#   - one 那一句必須與提示詞檔裡的原文逐字相同，prompt_text() 會檢查
CALL_RULE = {
    "one": "- Call exactly one tool per step. Wait for the result before"
           " deciding\n  the next step.",
    "batch": "- You may call more than one tool in a step. They run in the"
             " order you\n  list them, and you get every result back.",
}


def prompt_text(role: str, root: Path = DATA, tool_calls: str = "one") -> str:
    """prompt_text（讀某個角色的提示詞檔），照模式換掉呼叫規則那一行。"""
    p = root / "prompts" / f"{role}.txt"
    if not p.exists():
        raise ValueError(f"沒有 {role} 的提示詞檔：{p}")
    text = p.read_text(encoding="utf-8").rstrip() + "\n"
    if tool_calls not in CALL_RULE:
        raise ValueError(f"不認得的呼叫模式 {tool_calls!r}")
    if role not in ROLES:
        return text                        # 不呼叫工具的角色，沒有這一行
    if CALL_RULE["one"] not in text:
        raise ValueError(f"{role} 的提示詞裡找不到呼叫規則那一行，"
                         f"CALL_RULE['one'] 要跟著提示詞檔改")
    return text.replace(CALL_RULE["one"], CALL_RULE[tool_calls])


def all_prompts(root: Path = DATA) -> dict:
    """all_prompts（所有角色的提示詞）。"""
    return {r: prompt_text(r, root) for r in ROLES}


def _sig(spec: dict) -> str:
    """_sig（signature，一個參數在工具清單裡寫成什麼樣子）。
    有合法值清單（enum）的參數列出合法值，其餘只寫型別，
    跟原生函式呼叫送出去的工具定義一致。
    """
    if spec.get("enum"):
        return "|".join(f'"{v}"' for v in spec["enum"])
    return str(spec.get("type", "string"))


def tool_cards_block(tools: dict, role: str) -> str:
    """tool_cards_block（提示詞尾端那段工具清單）。
    把這個角色有權限用的工具一次全部列出來，每支一行：
    工具名、參數（含合法值）、一句描述。

    只放這三樣。實作路徑、前置條件、副作用清單都不放。
    """
    lines = ["", "Available tools:"]
    for tid in sorted(tools):
        t = tools[tid]
        if role not in (t.get("permissions") or []):
            continue
        params = ", ".join(
            f"{n}: {_sig(s)}" + ("" if s.get("required") else "?")
            for n, s in (t.get("params") or {}).items())
        lines.append(f"- {tid}({params}) — {t.get('description', '')}")
    return "\n".join(lines) + "\n"


def system_prompt(role: str, tools: dict, root: Path = DATA,
                  tool_calls: str = "one") -> str:
    """system_prompt（模型真正看到的第一段文字）＝ 提示詞 ＋ 工具清單。"""
    return (prompt_text(role, root, tool_calls)
            + tool_cards_block(tools, role))
