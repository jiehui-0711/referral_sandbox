"""載入醫療指引：條件式文件，一條規則一份。
指引是 Agent 做臨床決定的唯一依據，而且它是條件式的：

    when:   {procedure_category: endoscopic, drug: warfarin,
             indication_class: afib_nonvalvular, procedure_risk: high}
    action: {hold_days: 5, substitute: none, resume_days_after: 1}

  - 一個任務讀得到哪些指引，由它的 elements（元素清單）決定
  - 回傳給 Agent 的欄位只有五個：
    guideline_id / decision / title / when / action
  - assumption / source / reviewed_by 不回傳
"""
from __future__ import annotations

from pathlib import Path

from ..paths import DATA, load_yaml, yaml_files

# AGENT_FIELDS：Agent 讀得到的欄位。這張清單就是檢索索引的範圍。
AGENT_FIELDS = ("guideline_id", "decision", "title", "when", "action")

# NEVER_RETURN：給人看的設計說明，絕對不回傳。
NEVER_RETURN = ("assumption", "source", "source_type", "reviewed_by")

REQUIRED = ("guideline_id", "decision", "title", "when", "action")


def load_guidelines(root: Path = DATA) -> dict:
    """load_guidelines（把元素庫裡所有的指引讀進來），回傳 {指引代號: 指引}。
    這裡載入全部，由 build_world 依任務的元素清單篩選。
    """
    out: dict = {}
    folder = root / "elements"
    if not folder.is_dir():
        return out
    for el in sorted(p for p in folder.iterdir() if p.is_dir()):
        g_dir = el / "guidelines"
        if not g_dir.is_dir():
            continue
        for p in yaml_files(g_dir):
            d = load_yaml(p)
            gid = d["guideline_id"]
            if gid != p.stem:
                raise ValueError(
                    f"{p}：guideline_id 寫 {gid!r}，但檔名是 {p.stem!r}")
            if gid in out:
                raise ValueError(f"指引 {gid!r} 定義了兩次")
            missing = [k for k in REQUIRED if k not in d]
            if missing:
                raise ValueError(f"{p}：缺少欄位 {missing}")
            if not isinstance(d["when"], dict) or not d["when"]:
                raise ValueError(f"{p}：when 必須是非空的條件")
            if not isinstance(d["action"], dict) or not d["action"]:
                raise ValueError(f"{p}：action 必須是非空的結論")
            d["element"] = el.name
            out[gid] = d
    return out


def for_agent(guideline: dict) -> dict:
    """for_agent（這一份指引裡 Agent 讀得到的部分）。
    用白名單：以後多加一個設計說明欄位，它預設就是看不到的。
    """
    return {k: guideline[k] for k in AGENT_FIELDS if k in guideline}


def by_decision(guidelines: dict) -> dict:
    """by_decision（依「它管哪一個決定」分組），回 {決定: [指引…]}。
      - 給定一組條件，同一個決定只對應一份指引
      - 零份也可以，那代表正確答案是上報醫師
      - 兩份以上適用同一組條件就是資料錯誤
    """
    out: dict = {}
    for gid in sorted(guidelines):
        out.setdefault(guidelines[gid]["decision"], []).append(guidelines[gid])
    return out
