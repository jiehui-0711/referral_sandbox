"""把工具定義翻譯成模型看得懂的格式。
「function calling」（函式呼叫）是模型原生支援的一種介面：
我們告訴模型「有這些工具、每個工具收哪些參數」，
模型就能回一個結構化的呼叫請求，而不是吐一段要我們自己剖析的文字。

  - 一律用模型原生的那套，不用請你輸出 JSON 那種文字約定
  - 巢狀結構一定要展開宣告：有些參數是清單或物件，
    不宣告裡面該放什麼，模型就只能猜
"""
from __future__ import annotations

# JSON_TYPE：沙盒內部的型別名稱 -> 模型那邊用的型別名稱
JSON_TYPE = {"string": "string", "int": "integer", "bool": "boolean",
             "date": "string", "list": "array", "object": "object"}

# DOT：工具名裡如果有點號，要換成這個再送給模型（點號在那邊有別的意思）。
# 目前的工具名都沒有點號，但之後工具變多、需要分群時會有。
DOT = "__"


def to_function_name(tool_id: str) -> str:
    """to_function_name（工具名 -> 送給模型的名字）。"""
    return tool_id.replace(".", DOT)


def from_function_name(name: str) -> str:
    """from_function_name（模型回來的名字 -> 工具名）。"""
    return str(name).replace(DOT, ".")


def param_schema(spec: dict) -> dict:
    """param_schema（一個參數的定義 -> 模型看得懂的宣告）。
    物件與清單會往裡面遞迴展開，不只宣告這是個清單。
    """
    t = spec.get("type", "string")
    out: dict = {"type": JSON_TYPE.get(t, "string")}
    desc = []
    if t == "date":
        desc.append("date as YYYY-MM-DD")
    if spec.get("description"):
        desc.append(str(spec["description"]))
    if desc:
        out["description"] = " ".join(desc)
    if spec.get("enum"):
        out["enum"] = list(spec["enum"])
    if t == "object":
        props = spec.get("properties") or {}
        out["properties"] = {k: param_schema(v) for k, v in props.items()}
        req = [k for k, v in props.items() if v.get("required")]
        if req:
            out["required"] = req
    if t == "list":
        out["items"] = param_schema(spec["items"]) if spec.get("items") else {}
    return out


def tool_schema(tool: dict) -> dict:
    """tool_schema（一支工具的完整宣告：名字、描述、參數）。"""
    params = tool.get("params") or {}
    return {"type": "function", "function": {
        "name": to_function_name(tool["tool_id"]),
        "description": tool.get("description", ""),
        "parameters": {
            "type": "object",
            "properties": {n: param_schema(p) for n, p in params.items()},
            "required": [n for n, p in params.items() if p.get("required")],
        },
    }}


def all_tool_schemas(tools: dict, role: str = "referral_agent") -> list[dict]:
    """all_tool_schemas（這個角色能用的所有工具的宣告）。
    目前一次把全部列出來，不做工具搜尋。
    """
    return [tool_schema(tools[tid]) for tid in sorted(tools)
            if role in (tools[tid].get("permissions") or [])]
