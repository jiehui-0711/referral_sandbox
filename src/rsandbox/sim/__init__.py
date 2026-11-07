"""模擬代理人（sim ＝ simulated person，世界裡的人）。
這個資料夾不是在設定不同身份的 Agent。
被測的那個 Agent 只有一個，它在 agents/react.py。
這裡是世界裡的人，目前只有病人。它跟診次、指引一樣是環境的一部分，
只是剛好會說話。

    patient.py   病人。由一個 LLM 扮演，讀情境的 patient_doc.yaml
    boundary.py  模擬代理人的越線紀錄

病人的生死線：絕對不主動提示。它絕不問那我的藥要不要跟著改。
模擬器一開口就替 Agent 完成了偵測這一步，而且什麼都不會壞。
規則寫在它自己的提示詞（data/prompts/patient_agent.txt）；要確認它有
沒有越線，讀 contact_log 裡它實際說過的每一句話。

醫師不在這裡：它只在停藥醫囑上蓋一個固定的章，沒有行為、不說話，
所以那個章直接寫在 handlers/clinic.py 的 issue_hold_order 裡。
"""
from . import boundary, patient
from .patient import answer as patient_answer

__all__ = ["boundary", "patient", "patient_answer"]
