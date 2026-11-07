"""載入病人的個人視角資料。
  - 來源：data/cases/<組>/patient_doc.yaml，一份第一人稱散文，餵給病人 LLM
  - 世界裡只有病人一個模擬代理人
  - 醫師的簽章是一個固定值，寫在 handlers/clinic.py
"""
from __future__ import annotations

from pathlib import Path

from ..paths import DATA, load_yaml

# PATIENT_KEYS：個人視角資料允許出現的欄位。多打一個會載入失敗。
PATIENT_KEYS = frozenset({"patient_id", "persona", "knows", "log"})


def load_patient_docs(root: Path = DATA) -> dict:
    """load_patient_docs（把每一組情境的病人資料讀進來），回傳 {病人代號: 資料}。"""
    out: dict = {}
    for p in sorted((root / "cases").glob("*/patient_doc.yaml")):
        d = load_yaml(p)
        unknown = set(d) - PATIENT_KEYS
        if unknown:
            raise ValueError(f"{p}：不認得的欄位 {sorted(unknown)}")
        pid = d["patient_id"]
        if pid in out:
            raise ValueError(f"病人 {pid!r} 的個人資料出現在兩組情境裡")
        if not str(d.get("persona") or "").strip():
            raise ValueError(f"{p}：缺少 persona")
        if not (d.get("knows") or {}):
            raise ValueError(f"{p}：knows 不可為空")
        for name, text in d["knows"].items():
            if not str(text or "").strip():
                raise ValueError(f"{p}：knows.{name} 是空的")
        for i, e in enumerate(d.get("log") or []):
            if "date" not in e or "text" not in e:
                raise ValueError(f"{p}：log[{i}] 要有 date 與 text")
        out[pid] = d
    if not out:
        raise ValueError(f"一份病人資料都沒有：{root / 'cases'}/*/patient_doc.yaml")
    return out
