"""信封（envelope）：所有工具統一的回傳格式，以及合法失敗理由的全表。
每一支工具都回傳同一種結構：

    {"ok": bool, "reason": str | None, "data": dict | None}

  - 成功（ok=true）：reason 必為空，data 裝回傳內容
  - 失敗（ok=false）：reason 必須是 REASONS 裡的一個，data 必為空

REASONS 裡沒有任何一個理由跟臨床正確性有關。工具不會因為以下理由拒絕：
  - 這個日期對病人不好
  - 停藥天數不夠
  - 這是門診診次，不能做大腸鏡
  - 這個病人在吃抗凝血藥

什麼時候回信封、什麼時候丟例外：
  - 世界的合法結果（號滿了、查無資料）：handler 回失敗信封，原樣交給 Agent
  - 呼叫方式錯誤（型別錯、必填漏掉）：handler 丟 ToolError，攔下來轉成信封
  - 資料檔壞掉（YAML 格式錯）：丟 DataError，中止整個 episode

Agent 永遠只看得到信封，不會看到例外。
"""
from __future__ import annotations

# 失敗理由全表。新增一個理由的判準：
#   - 說「做不到」的可以加：號滿了、這一科不存在、你不在那個地方
#   - 說「不該做」的不可以加：那是臨床判斷
REASONS = frozenset({
    #  呼叫方式錯了
    "bad_arguments",        # 型別錯、必填漏掉、值不在允許清單裡、日期格式不合
    "not_found",            # 參數格式對，但查不到這筆資料
    "permission_denied",    # 角色權限不足。目前只有一個角色，不會發生

    #  位置不對
    "not_available_here",   # 這支工具不能在你現在站的地方用。
                            #   訊息固定，不提示該去哪裡
    "not_adjacent",         # 要移動過去，但那兩個節點之間沒有路

    #  世界裡查無此物
    "unknown_institution",  # 沒有這間機構
    "unknown_department",   # 沒有這一科。跟「科存在但沒公布資訊」不同
    "unknown_service",      # 這一科沒有這個服務項目
    "unknown_slot",         # 沒有這個診次代號

    #  診次的狀態
    "capacity_full",        # 已訂人數已經達到這個診次的容量上限
    "slot_unavailable",     # 診次已停診，或日期已經過了

    #  醫囑與預約的狀態
    "already_superseded",   # 想取代的那張單子已經不是生效中的了
    "already_completed",    # 已經抽過的血不能撤銷
    "no_active_appointment",# 要改期，但那筆預約已經不是「已訂」狀態

    #  行政前置條件沒滿足
    "not_referable",        # 這間機構不能被轉介（C0 是 Agent 自己的診所）
    "prerequisite_missing", # 這家醫院規定訂號前要有轉診，而這個案子沒有。
                            #   純行政：它不檢查轉診是不是轉去對的科
})


class DataError(RuntimeError):
    """DataError（資料檔壞掉）：中止整個 episode，標記無效，不計入結果。"""


class ToolError(Exception):
    """ToolError（呼叫錯了，或世界的合法拒絕）。會被攔下來轉成失敗信封。
      - reason 必須在 REASONS 裡
      - detail 只給開發者看，不會進到 Agent 看得到的信封裡
      - fields 會進到信封的 fields 欄位，Agent 讀得到。裝的是哪幾個參數
        有問題、問題是什麼，不裝傳進來的值、也不裝合法值清單
    """

    def __init__(self, reason: str, detail: str = "",
                 fields: list | None = None) -> None:
        if reason not in REASONS:
            raise DataError(f"{reason!r} 不在 REASONS 裡")
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail
        self.fields = list(fields or [])


def ok(data: dict | None = None) -> dict:
    """ok（裝一個成功信封）。"""
    return {"ok": True, "reason": None, "data": dict(data or {})}


def fail(reason: str, fields: list | None = None) -> dict:
    """fail（裝一個失敗信封）。理由不在表上就直接當成資料錯誤丟出來。
    fields 只在有東西說的時候才出現在信封裡，裝的是哪幾個參數有問題。
      - 只放欄位名與問題類型，不放傳進來的值、不放合法值清單
      - reason 是封閉的 16 個字串，所以欄位名沒辦法塞進那裡
    """
    if reason not in REASONS:
        raise DataError(f"{reason!r} 不在 REASONS 裡")
    env = {"ok": False, "reason": reason, "data": None}
    if fields:
        env["fields"] = list(fields)
    return env
