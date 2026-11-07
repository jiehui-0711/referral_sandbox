"""指引檢索：BM25、回傳的欄位、以及「撈得到」這件事。
這一段不是研究重點。指引是條件式的，所以難度不在讀懂文字，
  而在從撈回來的幾份裡挑對哪一份適用。檢索只是把候選送到眼前的
  必要機制，目標是讓它幾乎不出錯。

  所以這個檔案要守的是兩件事：
        - 該撈到的一定撈得到（不然任務根本做不完，而且看起來像推理失敗）
        - 不該給的一個字都不給（設計說明寫中文，回出去等於送答案）
"""
from __future__ import annotations

import unittest

from rsandbox.loaders import load_all
from rsandbox.loaders.guidelines import AGENT_FIELDS, for_agent
from rsandbox.runtime.retrieval import DEFAULT_TOP_K, MAX_TOP_K, Index
from rsandbox.runtime.session import Session

DATA = load_all()
CASE = "COLO-01"
TASK = f"{CASE}/v0_clean"


def fresh():
    return Session(DATA, DATA["tasks"][TASK])


def search(query, top_k=None):
    args = {"query": query}
    if top_k is not None:
        args["top_k"] = top_k
    env = fresh().call("search_guidelines", args)
    assert env["ok"], env
    return env["data"]["results"]


class TestWhatComesBack(unittest.TestCase):

    def test_only_the_five_agent_fields_come_back(self):
        """白名單，不是黑名單。以後多加一個設計說明欄位，
        它預設就是看不到的 —— 不必記得把它加進禁止清單。"""
        for row in search("warfarin"):
            self.assertEqual(set(row), set(AGENT_FIELDS))

    def test_the_design_notes_never_leak(self):
        """`assumption` 寫中文，而且有「本案例命中這一條」這種句子。"""
        for q in ("warfarin", "colonoscopy", "blood test", "xyzzy 不相干", ""):
            blob = str(search(q))
            for forbidden in ("assumption", "source", "reviewed_by", "element",
                              "本案例"):
                self.assertNotIn(forbidden, blob, f"查 {q!r} 夾帶了 {forbidden}")

    def test_no_score_is_returned(self):
        """回了分數，Agent 就直接挑第一名 —— 那等於工具替它做了
        「哪一份適用」的判斷，而那正是要測的能力。"""
        for row in search("warfarin high bleeding risk"):
            for k in row:
                self.assertNotIn("score", str(k).lower())

    def test_an_off_target_query_still_returns_a_full_page(self):
        """回空清單本身就是回饋訊號，等於告訴 Agent「你查錯方向」。"""
        self.assertEqual(len(search("xyzzy 完全不相干的東西")), DEFAULT_TOP_K)
        self.assertEqual(len(search("")), DEFAULT_TOP_K)


class TestTheIndexIsWhatTheAgentSees(unittest.TestCase):
    """索引的內容 = Agent 讀得到的內容，一字不差。
    以前有一個 `keywords` 欄位只進索引、不回傳 ——
    那等於偷偷替正確答案加分，而且實測過它真的在作用：
    把它從索引拿掉之後，主案例查詢的第一名就翻掉了。

    現在索引吃的就是「要回傳的那個字典」，所以兩邊不可能分岔。
    """

    def test_the_index_is_built_from_the_returned_dict(self):
        g = DATA["worlds"][CASE]["guidelines"]
        visible = {gid: for_agent(doc) for gid, doc in g.items()}
        index = Index(visible)
        # 索引裡的每一個詞，都必須能在「回傳的內容」裡找到
        for gid in index.ids:
            text = str(visible[gid]).lower()
            for word in index.tokens[gid]:
                self.assertIn(word, text, f"{gid} 的索引有 Agent 看不到的詞 {word}")

    def test_no_guideline_has_a_field_outside_the_whitelist_in_the_index(self):
        """反面：確認指引檔案裡真的有不回傳的欄位。
        沒有的話，上面那條就是空測試。"""
        g = DATA["guidelines"]["HOLD-01"]
        self.assertIn("assumption", g)
        self.assertNotIn("assumption", for_agent(g))


class TestEverythingNeededIsReachable(unittest.TestCase):
    """最要緊的一組：該撈到的一定撈得到。
    設太小的風險不是「它讀錯」而是「它永遠撈不到」——
    那會讓任務無法完成，而且看起來像推理失敗。
    """

    # 主案例的決定鏈：處置 → 出血風險 → 停藥 → 術前抽血
    CHAIN = [
        ("which procedure after a positive bowel screening test "
         "no alarm features", "PROC-01"),
        ("bleeding risk of a full colonoscopy endoscopic procedure", "RISK-01"),
        ("withholding warfarin before a high bleeding risk endoscopic "
         "procedure non-valvular atrial fibrillation", "HOLD-01"),
        ("pre-procedure blood test when an anticoagulant has been withheld",
         "LAB-01"),
    ]

    def test_every_step_of_the_chain_is_retrieved(self):
        for query, want in self.CHAIN:
            with self.subTest(want=want):
                ids = [r["guideline_id"] for r in search(query)]
                self.assertIn(want, ids, f"查不到 {want} —— 任務做不完")

    def test_the_default_page_is_bigger_than_one_decision_group(self):
        """預設筆數必須大於同一個決定底下的份數。
        停藥有四份。如果預設筆數小於四，就可能撈回三份都不吻合的，
        而正確那一份被擠掉了。
        """
        from rsandbox.loaders.guidelines import by_decision
        biggest = max(len(v) for v in by_decision(DATA["guidelines"]).values())
        self.assertGreater(DEFAULT_TOP_K, biggest,
                           f"同一個決定底下最多 {biggest} 份，"
                           f"預設只回 {DEFAULT_TOP_K} 筆太少")

    def test_the_whole_decision_group_comes_back_together(self):
        """查停藥要把四份都撈回來，Agent 才有東西可以挑。"""
        ids = set(r["guideline_id"] for r in
                  search("withholding warfarin aspirin before an endoscopic "
                         "procedure", top_k=MAX_TOP_K))
        for gid in ("HOLD-01", "HOLD-02", "HOLD-03", "HOLD-04"):
            self.assertIn(gid, ids, gid)


class TestItIsDeterministic(unittest.TestCase):

    def test_the_same_query_always_gives_the_same_order(self):
        """整批實驗可重現的前提。平手時用指引代號決勝。"""
        for q in ("warfarin", "colonoscopy", "xyzzy"):
            a = [r["guideline_id"] for r in search(q)]
            b = [r["guideline_id"] for r in search(q)]
            self.assertEqual(a, b, q)

    def test_top_k_is_bounded(self):
        """上限存在：再大就等於把整個指引庫倒給它，
        那樣「挑對哪一份」就不用挑了。"""
        env = fresh().call("search_guidelines",
                           {"query": "x", "top_k": MAX_TOP_K + 1})
        self.assertFalse(env["ok"])
        self.assertEqual(env["reason"], "bad_arguments")


class TestTheLureIsAlwaysInThePage(unittest.TestCase):
    """誘餌必須跟正解一起回來，否則「挑對哪一份」不用挑。"""

    def test_the_mechanical_valve_guideline_comes_back_too(self):
        got = {r["guideline_id"]: r for r in
               search("withholding warfarin before an endoscopic procedure")}
        self.assertIn("HOLD-01", got)
        self.assertIn("HOLD-02", got)
        # 兩份宣稱管同一個決定，結論不同，只有條件分得出來
        self.assertEqual(got["HOLD-01"]["decision"], got["HOLD-02"]["decision"])
        self.assertNotEqual(got["HOLD-01"]["action"], got["HOLD-02"]["action"])
        self.assertNotEqual(got["HOLD-01"]["when"], got["HOLD-02"]["when"])

    def test_the_environment_does_not_say_which_one_applies(self):
        """不標註、不排除、不解釋。挑錯的錯誤要一路寫下去。"""
        for row in search("withholding warfarin"):
            blob = str(row).lower()
            for hint in ("applies", "applicable", "recommended", "match",
                         "correct", "best"):
                self.assertNotIn(hint, blob, hint)


if __name__ == "__main__":
    unittest.main()
