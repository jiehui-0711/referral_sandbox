"""載入病歷。病歷住在各組任務自己的資料夾裡：
    data/cases/COLO-01/record.yaml
    data/cases/TKA-01/record.yaml
這個檔只提供兩個入口：
    seed()        唯讀的種子，避免資料庫裡的檔案被修改
    fresh_copy()  每個 episode 一份新的，給任務執行時讀取、修改
"""
from __future__ import annotations

import copy
from pathlib import Path

from ..paths import DATA, load_yaml

_seed: dict | None = None


def load_seed(root: Path = DATA) -> dict:
    """load_seed（把病歷讀進來）。
    回傳的是唯讀種子，不要直接改它，要改請用 fresh_copy()。
    """
    out: dict = {}
    for p in sorted((root / "cases").glob("*/record.yaml")):
        d = load_yaml(p)
        pid = d["patient_id"]
        if pid in out:
            raise ValueError(f"病人 {pid!r} 的病歷出現在兩組任務裡（{p}）")
        out[pid] = d
    if not out:
        raise ValueError(f"一份病歷都沒有：{root / 'cases'}/*/record.yaml")
    return out


def seed(root: Path = DATA) -> dict:
    """seed（拿到唯讀的種子）。只有真資料目錄會被快取起來重複使用。"""
    global _seed
    if _seed is None or root != DATA:
        s = load_seed(root)
        if root == DATA:
            _seed = s
        return s
    return _seed


def fresh_copy(root: Path = DATA) -> dict:
    """fresh_copy（每個 episode 專屬的可寫副本）。
    deepcopy 是連裡面每一層都複製一份，所以改這一份不會影響種子
    或別的 episode。
    """
    return copy.deepcopy(seed(root))
