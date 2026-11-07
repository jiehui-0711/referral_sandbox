"""資料檔放在哪裡，以及怎麼把 YAML 讀進來。
所有資料都在 src/rsandbox/data/ 底下。

每個載入器都收一個選填的 root（資料根目錄）參數，
所以可以指到別的資料目錄，不一定要用內建的那一份。
"""
from __future__ import annotations

from pathlib import Path

import yaml

DATA = Path(__file__).parent / "data"


def load_yaml(path: Path):
    """load_yaml（讀一個 YAML 檔）。"""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def yaml_files(folder: Path) -> list[Path]:
    """yaml_files（一個資料夾裡所有的 .yaml 檔）
    排序是為了讓載入順序固定，不隨檔案系統而變。
    """
    return sorted(p for p in folder.glob("*.yaml"))
