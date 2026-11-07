"""病歷相關的工具。"""
from __future__ import annotations


def get_patient_record(ctx):
    """get_patient_record（調閱病歷）：回傳一整份病歷。
    回的是原料，不是結論。病歷裡沒有這三樣，都要 Agent 自己推：
      - 用藥原因屬於哪一類
      - 這個處置算高風險還是低風險
      - 任何算好的日期
    靠病歷資料本身寫對來守，程式不檢查，所以這裡直接回傳整份。

    不收參數：一個案子只有一個病人，由任務檔的 patient_id 決定。
    """
    return {"record": ctx.records[ctx.patient_id]}
