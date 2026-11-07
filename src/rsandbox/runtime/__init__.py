"""執行期：Agent 呼叫工具、世界被改動、時間往前走的地方。
這個資料夾分三層，由下往上疊。看程式建議照這個順序。

第一層：無狀態的零件。誰都可以用，彼此不相依。

    clock.py      模擬時鐘。日期字串的加減、算星期幾
    ids.py        物件編號產生器（APT-001、HO-001 這種流水號）
    messages.py   環境會對 Agent 說的每一句話，集中成一張表
    retrieval.py  指引檢索引擎（BM25 演算法，自己實作的）
    casework.py   「哪一筆預約才算數」的唯一定義

第二層：一次工具呼叫怎麼走。讀程式的主幹在這裡。

    envelope.py   回傳格式 {"ok", "reason", "data"}，以及合法 reason 的白名單
    session.py    Session（一次執行的全部狀態）。唯一會改變世界的入口
    executor.py   四道關卡（工具存在 → 有權限 → 參數格式 → 行政前置條件）
    handlers/     真正動手改資料的地方，一支工具一支函式

第三層：一次 episode（一個案例從頭跑到尾）怎麼跑。

    initial.py    這一題要從什麼狀態開始
    loop.py       主迴圈：發事件 → 給觀察 → 喚醒 Agent → 推進時鐘
    changelog.py  每次呼叫前後的資料差異，只追加不修改
    runio.py      把一次執行的結果寫成檔案

一次工具呼叫的完整路徑：

    Agent
      -> Session.call(工具名, 參數)                    session.py
           -> 這個工具存在嗎                           bad_arguments
           -> validate_args（檢查參數格式）            executor.py
           -> check_preconditions（檢查前置條件）      executor.py
           -> handler（真正改動資料）                  handlers/*.py
      <- {"ok", "reason", "data"}  同時寫一筆呼叫紀錄

沒有第五道關卡。工具永遠不會因為臨床上不適當而擋下呼叫：
傳一個跟預約日對不起來的停藥日期，照樣 ok=true。
對錯不在這裡判定，留在最終資料與呼叫紀錄裡。
"""
from .envelope import REASONS, DataError, ToolError, fail, ok
from .session import Session

__all__ = ["Session", "REASONS", "ToolError", "DataError", "ok", "fail"]
