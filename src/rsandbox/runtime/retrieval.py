"""BM25 檢索：Agent 怎麼在指引庫裡找到該用的那幾份。
  - 自己寫而不是裝一個套件，為的是確定性：同一個查詢在不同機器、
    不同版本上回完全一樣的順序。這裡沒有任何隨機來源，分數平手時
    一律用指引代號決勝
  - 索引的內容 = Agent 讀得到的內容，一字不差。做法是直接把要回傳給
    Agent 的那個字典攤平成文字來建索引，見 _text
  - 分數不回傳。排名順序本身已經是一種引導，但至少不要再加上一個
    這份有多相關的數字
  - 這一段不是研究重點：難度在挑對哪一份適用，不在讀懂文字，
    所以回傳筆數要寬鬆，見 DEFAULT_TOP_K
"""
from __future__ import annotations

import math
import re
from collections import Counter

# 英文／數字詞，以及單一中日韓字。底線會被切開，所以 FIT_positive
# 會變成 ["fit", "positive"]，用 "FIT positive" 查得到。
_TOKEN = re.compile(r"[A-Za-z]+|[0-9]+|[一-鿿]")

# DEFAULT_TOP_K：預設回幾份。
# 它必須大於一條決定鏈所需的指引數。主案例的鏈是
# 處置 → 出血風險 → 停藥 → 術前抽血，四份；而同一個決定底下
# 還有看起來很像的其他份（停藥有四份）。
DEFAULT_TOP_K = 8

# MAX_TOP_K：上限。再大就等於把整個指引庫倒給它。
MAX_TOP_K = 20

_K1 = 1.5
_B = 0.75


def tokenize(text: str) -> list[str]:
    """tokenize（把一段文字切成詞）。英文照字詞切，中文一個字一個詞。
    指引全部是英文，但查詢可能夾雜中文（模型讀到中文病歷就可能
    用中文查）。中文切得出來，用錯語言就只是分數很低，不會程式爆掉。
    """
    return [m.group(0).lower() for m in _TOKEN.finditer(str(text or ""))]


def _flatten(value) -> str:
    """_flatten（把嵌套的條件攤平成一行文字）。"""
    if isinstance(value, dict):
        return " ".join(f"{k} {_flatten(v)}" for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return " ".join(_flatten(v) for v in value)
    return str(value)


def _text(visible: dict) -> str:
    """_text（這一份指引拿來建索引的文字）。
    參數就是要回傳給 Agent 的那個字典，所以索引的內容與 Agent
    讀得到的內容完全相同。
    """
    return _flatten(visible)


class Index:
    """指引庫的 BM25 索引。建一次，查很多次。
    純函數：同一份指引庫建出來的索引，查同一個查詢永遠回同一個順序。
    """

    def __init__(self, visible_docs: dict) -> None:
        # visible_docs: {指引代號: Agent 讀得到的那個字典}
        self.ids = sorted(visible_docs)
        self.tokens = {gid: tokenize(_text(visible_docs[gid])) for gid in self.ids}
        self.freq = {gid: Counter(t) for gid, t in self.tokens.items()}
        self.length = {gid: len(t) for gid, t in self.tokens.items()}
        n = len(self.ids) or 1
        self.avg = (sum(self.length.values()) / n) or 1.0
        df: Counter = Counter()
        for gid in self.ids:
            df.update(set(self.tokens[gid]))
        # BM25 的逆文件頻率。加 0.5 的平滑是標準做法，避免除以零。
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5))
                    for w, c in df.items()}

    def _score(self, gid: str, words: list[str]) -> float:
        f = self.freq[gid]
        dl = self.length[gid] or 1
        total = 0.0
        for w in words:
            if w not in f:
                continue
            tf = f[w]
            total += self.idf.get(w, 0.0) * tf * (_K1 + 1) / (
                tf + _K1 * (1 - _B + _B * dl / self.avg))
        return total

    def search(self, query: str, top_k: int) -> list[str]:
        """search（查詢），回指引代號的清單，最相關的在前。
          - 永遠回滿 top_k 筆（指引庫夠大的話），即使分數全是 0，
            因為回空清單等於告訴 Agent 它的查詢方向不對
          - 平手時用指引代號決勝
        """
        words = tokenize(query)
        scored = sorted(((-self._score(gid, words), gid) for gid in self.ids))
        return [gid for _s, gid in scored[:max(0, top_k)]]
