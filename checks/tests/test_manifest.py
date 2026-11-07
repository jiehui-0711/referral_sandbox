"""世界的組裝，以及病歷要自己去查這件事。
這個檔只測程式行為，不測資料集寫得對不對：
      - 同一組的幾個變體拿到的世界完全相同
      - 少選一個元素，世界真的會縮小（否則上面那條是空測試）
      - 檢索對整個指引庫開放，不會只搜這一題需要的
      - 第一個觀察只給病人編號，病歷要 Agent 自己呼叫工具去查

  這個檔原本還檢查「必要清單宣告的東西是否真的存在」。那屬於
    設計任務階段要確認的事（清單寫錯了程式照跑），已經刪掉。
    manifest 這個欄位還在任務檔裡，純粹當設計紀錄，程式不讀也不對帳。
"""
from __future__ import annotations

import unittest

from rsandbox.loaders import load_all
from rsandbox.runtime.session import Session

DATA = load_all()
TASKS = sorted(DATA["tasks"])
# 世界是組裝出來的，一組一份。這個檔只有一組，所以取它的。
WORLD = DATA["worlds"]["COLO-01"]


class TestTheVariantsOfOneCaseShareOneWorld(unittest.TestCase):
    """世界是組裝出來的：固定層 ＋ 這一組選的元素。
    所以「每個任務的世界都一樣」已經不成立了，而且是刻意的 ——
    不同組選不同元素。但同一組的幾個變體必須完全一樣：

        變體只改狀態與事件（世界補丁、病人補丁、事件清單），
        不改世界組成。

    元素清單寫在 case.yaml，整組共用，所以這一條在結構上就成立；
    這個測試是防止哪天有人讓變體也能改元素清單。
    """

    def test_every_variant_of_a_case_gets_the_same_world(self):
        from rsandbox.loaders.cases import cases_of
        for cid, variants in cases_of(DATA["tasks"]).items():
            ref = Session(DATA, DATA["tasks"][variants[0]])
            for tid in variants:
                s = Session(DATA, DATA["tasks"][tid])
                with self.subTest(case=cid, variant=tid):
                    for key in ("institutions", "departments", "services",
                                "blocks", "slots"):
                        self.assertEqual(sorted(s.world[key]),
                                         sorted(ref.world[key]),
                                         f"{tid} 的 world.{key} 跟同組的其他變體"
                                         f"不同 —— 變體不該改世界組成")
                    self.assertEqual(sorted(s.world['guidelines']),
                                     sorted(ref.world['guidelines']))
                    self.assertEqual(sorted(s.tools), sorted(ref.tools))

    def test_the_element_list_is_the_union_of_necessary_and_distractor(self):
        """世界用的元素 = 必要 ∪ 干擾，一個不多一個不少。"""
        from rsandbox.loaders.cases import world_elements
        for tid in TASKS:
            t = DATA["tasks"][tid]
            with self.subTest(task=tid):
                self.assertEqual(
                    DATA["worlds"][t["case_id"]]["_elements"], world_elements(t))

    def test_leaving_an_element_out_really_shrinks_the_world(self):
        """反面：如果組裝沒有真的在篩選，上面那些就是空測試。
        只選大腸鏡元素的話，骨科、牙科、復健的節點根本不存在 ——
        走過去會回「查無此科」，而不是走得進去但沒資訊。
        """
        from rsandbox.loaders import world as W
        els = DATA["elements"]
        small = W.build_world(element_ids=["COLO"], elements=els)
        full = DATA["worlds"]["COLO-01"]
        self.assertLess(len(small["departments"]), len(full["departments"]))
        self.assertIn("H1_GI", small["departments"])
        for gone in ("H1_ORTH", "H3_DENT", "H1_REHAB", "H1_CR"):
            self.assertNotIn(gone, small["departments"], gone)
        # 段落與診次也要跟著縮
        self.assertLess(len(small["blocks"]), len(full["blocks"]))
        self.assertLess(len(small["slots"]), len(full["slots"]))
        # 但固定層一定還在：五家醫院與全院通行的規定
        self.assertEqual(sorted(small["institutions"]),
                         sorted(full["institutions"]))
        for kept in ("H2.3", "H4.2", "C0.1"):
            self.assertIn(kept, small["blocks"], kept)

    def test_search_covers_guidelines_the_task_does_not_need(self):
        """檢索對這一題世界裡的整個指引庫開放，不會只搜它需要的那幾份。
        每一題都撈得到 HOLD-02（機械瓣膜），而那一份沒有一個任務需要它 ——
        它存在的理由是讓「推錯用藥原因分類」得到一個似是而非的答案。
        """
        for tid in TASKS:
            s = Session(DATA, DATA["tasks"][tid])
            env = s.call("search_guidelines",
                         {"query": "withholding warfarin before an endoscopic "
                                   "procedure", "top_k": 8})
            with self.subTest(task=tid):
                self.assertTrue(env["ok"], env)
                ids = [r["guideline_id"] for r in env["data"]["results"]]
                self.assertIn("HOLD-02", ids, "誘餌沒被撈回來")


class TestTheRecordIsFetchedNotHandedOver(unittest.TestCase):
    """觀察只給病人編號，Agent 自己呼叫 get_patient_record 去查。
    2026-10-13 曾經改成把病歷全文直接附在第一回合的觀察裡（理由是
      「叫出病歷」沒有判斷、不會做錯），當天改回來。三個理由：

        - 會混淆歸因。 病歷會被事件改動（TKA-X2 追加 HbA1c、
          TKA-X3 補上術後住址）。附全文的話，Agent 手上那份會悄悄過期，
          於是「它有沒有回去重讀」跟「它有沒有重算衍生值」變成兩個
          糾纏在一起的失敗。保持「每次都去查」，那條混淆就不存在。
        - 少一個觀察點。 執行紀錄裡「有沒有讀病歷」原本是乾淨的訊號，
          附在提示詞裡就看不到了。
        - 實際跑下來沒出過差錯 —— 它本來就不是瓶頸。
    """

    def test_the_first_observation_gives_only_the_patient_id(self):
        from rsandbox.agents.scripted import ScriptedAgent
        from rsandbox.runtime import loop
        task = DATA["tasks"]["COLO-01/v0_clean"]
        s = loop.run_task(DATA, task, ScriptedAgent(DATA["scripts"]["COLO-01/v0_clean"]["reference"]))
        first = s.state["observations"][0]["text"]
        self.assertIn("P001", first)
        # 病歷的內容一個字都不能出現在觀察裡
        for token in ("Atrial fibrillation", "Warfarin", "valve_status",
                      "alarm_features", "Patient record"):
            self.assertNotIn(token, first,
                             f"觀察裡出現了病歷內容 {token!r} —— "
                             f"病歷應該由 Agent 自己去查")

    def test_the_agent_has_to_call_the_tool(self):
        """反面：工具要在，而且真的查得到。"""
        self.assertIn("get_patient_record", DATA["tools"])
        s = Session(DATA, DATA["tasks"]["COLO-01/v0_clean"])
        env = s.call("get_patient_record")
        self.assertTrue(env["ok"])
        self.assertEqual(env["data"]["record"]["valve_status"], "none")


if __name__ == "__main__":
    unittest.main()
