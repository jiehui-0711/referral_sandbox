"""被測 Agent：最單純的 ReAct 迴圈。
ReAct ＝ reasoning + acting（想一下、動一下、再想一下）。
它只是一個迴圈：把觀察送給模型 → 模型要求呼叫某個工具 → 執行 →
把結果送回去 → 重複，直到模型呼叫 end_turn。

刻意不做任何提示工程。以下這些一樣都沒有：

    範例示範（few-shot）
    要它一步一步想的指示（chain-of-thought）
    自我檢查或反省的回合
    規劃階段
    任何記得重算日期之類的提醒
    失敗自動重試的外層架構

系統提示詞就是 data/prompts/ 那個檔，加上工具清單，一字不多。

  - 用模型原生的函式呼叫介面，不用請輸出 JSON 那種文字約定
  - 刻意不提供 finished()（問它做完了沒）這個方法，主迴圈對它改用
    這回合有沒有寫入去推（見 runtime/loop.py）
  - 模型一則訊息給好幾個呼叫的時候怎麼辦有兩種模式，見 TOOL_CALL_MODES
"""
from __future__ import annotations

import copy
import json
import re

from ..loaders import prompts
from .backends import BackendError
from .schema import all_tool_schemas, from_function_name

# MAX_STEPS_PER_TURN（一個回合最多幾步）。
# 正確軌跡最長二十步左右，40 只是為了擋住模型在原地打轉。
MAX_STEPS_PER_TURN = 40

# NUDGE（推一下）：模型只講話、不呼叫工具時，送這一句提醒它。
#   - 它只重複提示詞已經說過的操作規則（一步呼叫一個工具、做完了呼叫
#     end_turn），所有任務都是同一句
#   - 推一次之後還是不呼叫工具，就當它沒事做
NUDGE = "Call a tool, or call end_turn if you are done."

# TOOL_CALL_MODES（模型一則訊息給好幾個呼叫的時候怎麼辦）。
#   - one    只執行第一個，其餘回 EXTRA_CALL，不執行
#   - batch  依序執行整批，每一個的結果都送回去
#   - 提示詞裡的呼叫規則跟著模式換，見 loaders/prompts.py 的 CALL_RULE
TOOL_CALL_MODES = ("one", "batch")

# EXTRA_CALL：被判為多餘的呼叫回這一則。
#   - one 模式：第二個以後的全部
#   - batch 模式：排在 end_turn 後面的那些
EXTRA_CALL = {"error": "Only one tool call per step. This call was not executed."}

# LOOKS_LIKE_A_CALL：偵測工具呼叫被寫成了文字。
# 有些模型在講話的內容裡寫了一段看起來像工具呼叫的 JSON，但原生的
# 呼叫欄位是空的：它想呼叫工具，那個呼叫卻留在文字裡，從未被執行。
# 原因通常在模型與服務端之間那層，服務端的剖析器要求呼叫必須在
# 回應的最開頭。
#   - 偵測、計數、寫進執行紀錄，但不執行
#   - 去文字裡把 JSON 撈出來執行就是退回文字約定，所以不做
#   - 這個數字高的時候，那一批結果不能當成推理能力的量測
LOOKS_LIKE_A_CALL = re.compile(
    r'\{\s*"(?:name|tool|function)"\s*:\s*"([A-Za-z_][A-Za-z0-9_.]*)"')


class ReActAgent:
    """ReActAgent（真正受測的那個 Agent）。"""

    name = "react"

    def __init__(self, backend, max_steps: int = MAX_STEPS_PER_TURN,
                 nudges: int = 1, tool_calls: str = "one") -> None:
        if tool_calls not in TOOL_CALL_MODES:
            raise ValueError(f"不認得的呼叫模式 {tool_calls!r}")
        self.backend = backend
        self.max_steps = max_steps
        self.nudges = nudges
        # 一則訊息給好幾個呼叫時怎麼辦。見 TOOL_CALL_MODES。
        self.tool_calls = tool_calls
        self.messages: list[dict] = []     # 跟模型來回的對話歷史
        self.transcript: list[dict] = []   # 給人看的逐字稿，會寫成檔案
        self.errors: list[str] = []        # 後端層級的錯誤（連不上、看不懂）
        self.steps = 0
        # 被寫成文字、因此從未被執行的呼叫。見上面 LOOKS_LIKE_A_CALL。
        self.unparsed: list[dict] = []
        # 同一則訊息裡第二個以後的呼叫會被丟掉（提示詞明文：一步一個），
        # 丟掉的那些也要計數。
        self.dropped: list[dict] = []

    # ---------------- 主迴圈的介面 ----------------

    def activate(self, session, observation: str) -> None:
        """activate（跑一個回合）。
        對話歷史是跨回合累積的：第二個回合看得到第一個回合做了什麼。
        """
        if not self.messages:
            self.messages.append({"role": "system",
                                  "content": prompts.system_prompt(
                                      "referral_agent", session.tools,
                                      tool_calls=self.tool_calls)})
        self.messages.append({"role": "user", "content": observation})
        self.transcript.append({"t": session.today, "observation": observation})

        tools = all_tool_schemas(session.tools)
        nudged = 0
        for _ in range(self.max_steps):
            try:
                msg = self.backend.chat(self.messages, tools)
            except BackendError as e:
                # 連不上、或回應看不懂：記下來就停，不當成模型沒事做。
                self.errors.append(str(e))
                self.transcript.append({"t": session.today, "backend_error": str(e)})
                return

            msg.setdefault("role", "assistant")
            msg.setdefault("tool_calls", [])
            calls = msg.get("tool_calls") or []
            # 先切出要執行的與多餘的，_plain() 要用執行數標 executed。
            runnable, extras = self._split(calls)
            self.messages.append(msg)
            self.transcript.append({"t": session.today,
                                    "message": _plain(msg, len(runnable))})

            # 講話內容裡有看起來像工具呼叫的東西 -> 記一筆。
            # 即使同一則訊息也有原生呼叫，還是要記。
            for name in LOOKS_LIKE_A_CALL.findall(str(msg.get("content") or "")):
                self.unparsed.append({
                    "t": session.today, "name": name,
                    "had_native_call": bool(calls),
                    "content": str(msg.get("content"))[:500]})
                self.transcript.append({
                    "t": session.today, "unparsed_tool_call": name,
                    "had_native_call": bool(calls)})

            if not calls:
                if nudged >= self.nudges:
                    return                 # 推過了還是不呼叫工具，當它沒事做
                nudged += 1
                self.messages.append({"role": "user", "content": NUDGE})
                # 推一下也是 Agent 看到的東西，所以要寫進逐字稿。
                self.transcript.append({"t": session.today, "nudge": NUDGE})
                continue

            for extra in extras:
                self._tool_reply(extra.get("id", ""), extra.get("name", ""), EXTRA_CALL)
                self.dropped.append({"t": session.today,
                                     "name": from_function_name(extra.get("name", "")),
                                     "after": from_function_name(
                                         runnable[-1].get("name", ""))})

            for call in runnable:
                if self._run_one(session, call):
                    return                 # end_turn 成功，這個回合結束

    # ---------------- 內部 ----------------

    def _split(self, calls: list) -> tuple[list, list]:
        """_split（一則訊息裡的呼叫分成兩堆），回 (要執行的, 多餘的)。
          - one  ：只執行第一個
          - batch：依序執行整批，但排在 end_turn 後面的到不了，算多餘
        """
        if self.tool_calls == "one":
            return calls[:1], calls[1:]
        for i, c in enumerate(calls):
            if from_function_name(c.get("name", "")) == "end_turn":
                return calls[:i + 1], calls[i + 1:]
        return list(calls), []

    def _run_one(self, session, call: dict) -> bool:
        """_run_one（執行一個呼叫），回傳這個回合是不是結束了。
          - 整批執行時每一個都走這裡，順序就是模型列出來的順序
          - 回傳 True 的條件：這一個是 end_turn 而且成功
        """
        tool_id = from_function_name(call.get("name", ""))
        args = call.get("arguments") or {}
        if not isinstance(args, dict):
            args = {}

        env = session.call(tool_id, args)
        self.steps += 1
        payload = env["data"] if env["ok"] else _error_payload(env)
        self._tool_reply(call.get("id", ""), tool_id, payload)
        #   - 工具結果也寫進逐字稿，這樣逐字稿自己就讀得懂，
        #     不用跟最終狀態對照著看
        #   - 必須深拷貝：handler 回傳的結構裡有些巢狀的值（例如一筆
        #     預約的改期歷史）跟狀態裡的是同一個物件，不拷貝的話，
        #     之後改期會讓逐字稿裡早就寫好的那一則跟著變
        self.transcript.append({"t": session.today, "tool": tool_id,
                                "args": copy.deepcopy(args), "ok": env["ok"],
                                "reason": env["reason"],
                                "result": copy.deepcopy(payload)})
        return tool_id == "end_turn" and env["ok"]

    def _tool_reply(self, call_id: str, name: str, payload) -> None:
        """_tool_reply（把工具的結果送回給模型）。
        失敗的信封原樣送回去，只有失敗理由，沒有給開發者看的細節。
        模型有機會自己修正再試一次，失敗一樣消耗步數預算，
        而且每一次失敗都進了呼叫紀錄。
        """
        self.messages.append({
            "role": "tool", "tool_call_id": call_id, "name": name,
            "content": json.dumps(payload, ensure_ascii=False, default=str)})


def _error_payload(env: dict) -> dict:
    """_error_payload（失敗的信封送回給模型的樣子）。
      - error：失敗理由，REASONS 裡的一個
      - fields：哪幾個參數有問題，只有 bad_arguments 會帶
    """
    out = {"error": env["reason"]}
    if env.get("fields"):
        out["fields"] = list(env["fields"])
    return out


def _plain(msg: dict, n_executed: int = 1) -> dict:
    """_plain（把一則訊息整理成逐字稿要存的形狀）。
      - executed 標出哪幾個呼叫真的跑了：前 n_executed 個
      - one 模式是 1，batch 模式是整批扣掉排在 end_turn 後面的
    """
    return {"role": msg.get("role"), "content": msg.get("content"),
            "tool_calls": [{"name": c.get("name"), "arguments": c.get("arguments"),
                            "executed": i < n_executed}
                           for i, c in enumerate(msg.get("tool_calls") or [])]}
