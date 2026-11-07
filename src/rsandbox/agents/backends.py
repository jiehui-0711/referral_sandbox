"""模型後端：把各家不同的對話介面包成同一個樣子。
沙盒要能換模型，而各家的介面格式不一樣：同一個工具呼叫的參數，
一家給的是結構化物件，另一家給的是一串 JSON 文字；有一家的工具結果
還必須帶回這是回答哪一個呼叫的代號。
這些差異全部關在這個檔裡，Agent 那邊只需要處理一種格式。

正規化訊息（canonical message，統一過的訊息）長這樣：

    {"role": "system"|"user",  "content": 文字}
    {"role": "assistant", "content": 文字,
     "tool_calls": [{"id": 代號, "name": 工具名, "arguments": 參數}]}
    {"role": "tool", "tool_call_id": 代號, "name": 工具名, "content": 結果}

每個後端負責兩件事：
    把這種統一格式翻成自己家的格式送出去
    把自己家回來的東西翻回這種統一格式
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

# DOT：工具名裡的點號要換掉再送出去，
# 多數介面的函式名只接受英數字、底線與減號。
DOT = "__"


def to_function_name(tool_id: str) -> str:
    """to_function_name（工具名 -> 送給模型的名字）。"""
    return tool_id.replace(".", DOT)


def from_function_name(name: str) -> str:
    """from_function_name（模型回來的名字 -> 工具名）。"""
    return str(name).replace(DOT, ".")


def _as_dict(arguments) -> dict:
    """_as_dict（把參數統一成字典）。
    有的後端給的是現成的字典，有的給一串 JSON 文字，要分開處理。
    文字剖析失敗就回空字典，不丟例外。
    """
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str) and arguments.strip():
        try:
            out = json.loads(arguments)
            return out if isinstance(out, dict) else {"_value": out}
        except json.JSONDecodeError:
            return {}
    return {}


class BackendError(RuntimeError):
    """BackendError（連不上模型，或回傳的東西完全看不懂）。
    這類錯誤是基礎設施故障，不能被當成模型表現不好。
    """


class Backend:
    """Backend（後端的共同介面）。每一家實作自己的 chat()。"""

    name = "backend"

    def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        """chat（送出對話歷史，拿回模型的一則回覆）。"""
        raise NotImplementedError

    def usage_rows(self) -> list[dict]:
        """usage_rows（每一次呼叫的送出量與服務端處理量），一次一列。
          - sent_tokens：送出量的粗估
          - prompt_eval_count：服務端回報它實際處理了幾個 token
          - num_ctx：這一輪設定的上限，None ＝ 沒設
        """
        return list(getattr(self, "usage", []))


def _sent_tokens(payload: dict) -> int:
    """_sent_tokens（這一次送出去大約幾個 token）。
    按 3.5 個字元一個 token 粗估，不是精確計數。
    """
    return int(len(json.dumps(payload, ensure_ascii=False)) / 3.5)


def _post_json(url: str, payload: dict, headers: dict, timeout: int) -> dict:
    """_post_json（送一個 JSON 請求，把 JSON 回應讀回來）。
    每一種傳輸失敗都轉成 BackendError，不讓例外往上穿。
    """
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:600]
        raise BackendError(f"{url} 回傳 HTTP {e.code}：{body}") from e
    except urllib.error.URLError as e:
        raise BackendError(f"連不到 {url}：{e.reason}") from e
    except json.JSONDecodeError as e:
        raise BackendError(f"{url} 的回應不是 JSON：{e}") from e
    except TimeoutError as e:
        # 讀取逾時。TimeoutError 與 URLError 都是 OSError 的子類但
        # 互不相屬，要單獨攔一次。
        raise BackendError(f"{url} 超過 {timeout} 秒沒有回應") from e
    except OSError as e:
        raise BackendError(f"{url} 傳輸失敗：{e}") from e


# ---------------------------------------------------------------- Ollama

class OllamaBackend(Backend):
    """Ollama 的 /api/chat。arguments 是物件，tool 訊息不需要 id。"""

    name = "ollama"

    def __init__(self, model: str, base_url: str = "http://localhost:11434",
                 temperature: float = 0.0, seed: int = 0, timeout: int = 600,
                 num_ctx: int | None = None) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.seed = seed
        self.timeout = timeout
        # num_ctx（context window 上限）。None ＝ 不送這個參數，
        # 由服務端決定上限；超過上限的部分服務端會切掉，不報錯。
        self.num_ctx = num_ctx
        self.usage: list[dict] = []

    def serialize(self, messages: list[dict]) -> list[dict]:
        """serialize（把統一格式的歷史翻成 Ollama 要的格式）。"""
        out = []
        for m in messages:
            if m["role"] == "assistant":
                a = {"role": "assistant", "content": m.get("content") or ""}
                if m.get("tool_calls"):
                    a["tool_calls"] = [{"function": {"name": to_function_name(c["name"]),
                                                     "arguments": c["arguments"]}}
                                       for c in m["tool_calls"]]
                out.append(a)
            elif m["role"] == "tool":
                out.append({"role": "tool", "content": m["content"],
                            "tool_name": to_function_name(m.get("name", ""))})
            else:
                out.append({"role": m["role"], "content": m["content"]})
        return out

    def chat(self, messages, tools):
        options = {"temperature": self.temperature, "seed": self.seed}
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx
        payload = {"model": self.model, "messages": self.serialize(messages),
                   "tools": tools, "stream": False, "options": options}
        data = _post_json(f"{self.base_url}/api/chat", payload, {}, self.timeout)
        # 送出量與服務端處理量，判斷截斷用，見 cli.py 的 _ctx_note。
        self.usage.append({"sent_tokens": _sent_tokens(payload),
                           "prompt_eval_count": data.get("prompt_eval_count"),
                           "num_ctx": self.num_ctx})
        if "message" not in data:
            raise BackendError(f"Ollama 回應沒有 message 欄位：{json.dumps(data)[:600]}")
        msg = data["message"] or {}
        calls = []
        for i, c in enumerate(msg.get("tool_calls") or []):
            fn = c.get("function") or {}
            calls.append({"id": c.get("id") or f"call_{i}",
                          "name": from_function_name(fn.get("name", "")),
                          "arguments": _as_dict(fn.get("arguments"))})
        return {"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls}


# ---------------------------------------------------------------- OpenAI 相容

class OpenAICompatBackend(Backend):
    """OpenAI 相容格式的後端。
    很多服務都走這個格式（自架的推論伺服器、各種轉接服務、OpenAI 本身），
    所以一個實作可以接很多地方。

    跟上面那家的兩個差別：
      - 參數是一串 JSON 文字，不是物件
      - 工具結果必須帶回這是回答哪一個呼叫的代號
    """

    name = "openai"

    def __init__(self, model: str, base_url: str = "https://api.openai.com",
                 api_key: str | None = None, temperature: float = 0.0,
                 seed: int = 0, timeout: int = 600,
                 num_ctx: int | None = None) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.seed = seed
        self.timeout = timeout
        # 這個格式沒有 num_ctx 參數，收下來只記進 usage，
        # 讓兩家後端的紀錄形狀一樣。
        self.num_ctx = num_ctx
        self.usage: list[dict] = []

    def serialize(self, messages: list[dict]) -> list[dict]:
        """serialize（把統一格式的歷史翻成這家要的格式）。"""
        out = []
        for m in messages:
            if m["role"] == "assistant":
                a = {"role": "assistant", "content": m.get("content") or None}
                if m.get("tool_calls"):
                    a["tool_calls"] = [
                        {"id": c["id"], "type": "function",
                         "function": {"name": to_function_name(c["name"]),
                                      "arguments": json.dumps(c["arguments"], ensure_ascii=False)}}
                        for c in m["tool_calls"]]
                out.append(a)
            elif m["role"] == "tool":
                out.append({"role": "tool", "tool_call_id": m.get("tool_call_id", ""),
                            "content": m["content"]})
            else:
                out.append({"role": m["role"], "content": m["content"]})
        return out

    def chat(self, messages, tools):
        payload = {"model": self.model, "messages": self.serialize(messages),
                   "tools": tools, "tool_choice": "auto",
                   "temperature": self.temperature, "seed": self.seed}
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        data = _post_json(f"{self.base_url}/v1/chat/completions", payload, headers, self.timeout)
        self.usage.append({"sent_tokens": _sent_tokens(payload),
                           "prompt_eval_count": (data.get("usage") or {}
                                                 ).get("prompt_tokens"),
                           "num_ctx": self.num_ctx})
        choices = data.get("choices") or []
        if not choices:
            raise BackendError(f"回應沒有 choices：{json.dumps(data)[:600]}")
        msg = choices[0].get("message") or {}
        calls = []
        for i, c in enumerate(msg.get("tool_calls") or []):
            fn = c.get("function") or {}
            calls.append({"id": c.get("id") or f"call_{i}",
                          "name": from_function_name(fn.get("name", "")),
                          "arguments": _as_dict(fn.get("arguments"))})
        return {"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls}


# ---------------------------------------------------------------- 錄播

class ReplayBackend(Backend):
    """錄播後端：照事先寫好的腳本回覆，完全不連網路。
    它的用途不是模擬模型，而是把 Agent 那一段程式路徑走完：
    訊息怎麼串、工具呼叫怎麼解、錯誤怎麼餵回去、怎麼收尾。

    腳本的每一項是「這一輪要回什麼」：
        ("call", 工具名, 參數)   回一個工具呼叫
        ("say", 文字)            只講話、不呼叫工具
        ("raw", 整則訊息)        直接給一則訊息（用來測畸形的輸入）

    腳本用完之後一律回 end_turn，避免無限等下去。
    """

    name = "replay"

    def __init__(self, script: list) -> None:
        self.script = list(script)
        self.i = 0
        self.seen: list[dict] = []          # 收到的歷史，可以回頭檢查

    def chat(self, messages, tools):
        self.seen.append({"messages": [dict(m) for m in messages],
                          "tool_names": [t["function"]["name"] for t in tools]})
        if self.i >= len(self.script):
            step = ("call", "end_turn", {})
        else:
            step = self.script[self.i]
            self.i += 1
        kind = step[0]
        if kind == "say":
            return {"role": "assistant", "content": step[1], "tool_calls": []}
        if kind == "raw":
            return step[1]
        _, tool_id, args = step
        return {"role": "assistant", "content": "",
                "tool_calls": [{"id": f"call_{self.i}", "name": tool_id, "arguments": args}]}


def build_backend(provider: str, model: str, base_url: str | None = None,
                  api_key: str | None = None, seed: int = 0,
                  temperature: float = 0.0,
                  num_ctx: int | None = None) -> Backend:
    """build_backend（照指定的供應商名稱，做出對應的後端）。
    溫度預設 0、亂數種子固定，盡可能讓同一次執行可重現。
    模型端無法保證完全確定。
    """
    if provider == "ollama":
        return OllamaBackend(model, base_url or "http://localhost:11434",
                             temperature=temperature, seed=seed, num_ctx=num_ctx)
    if provider == "openai":
        return OpenAICompatBackend(model, base_url or "https://api.openai.com",
                                   api_key=api_key, temperature=temperature,
                                   seed=seed, num_ctx=num_ctx)
    raise ValueError(f"unknown provider {provider!r}")
