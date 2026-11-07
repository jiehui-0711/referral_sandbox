"""模擬代理人的生死線，以及停藥醫囑上那個無條件的簽章。
世界裡只有病人一個模擬代理人。它由一個 LLM 扮演，行為規則寫在
  data/prompts/patient_agent.txt。要確認它有沒有越線，讀 contact_log
  裡它實際說過的每一句話 —— 程式層沒有規則判斷。

醫師不是模擬代理人：它只在停藥醫囑上蓋一個固定的章，沒有行為、
  不說話。所以這裡測的不是「醫師會不會抓漏」，是「那個章真的無條件」。
"""
from __future__ import annotations

import unittest

from rsandbox.agents.scripted import ScriptedAgent
from rsandbox.loaders import load_all
from rsandbox.runtime import initial, loop
from rsandbox.runtime.handlers.clinic import AUTHORISER
from rsandbox.runtime.messages import render
from rsandbox.sim import boundary

DATA = load_all()
CASE = "COLO-01"
BASE = initial.clean_end(DATA, CASE)


def run(task_id, steps):
    task = DATA["tasks"][task_id]
    agent = ScriptedAgent(steps)
    s = loop.run_task(DATA, task, agent, BASE)
    return s, agent


def script(task_id: str, alt: str | None = None):
    """script（拿某個變體的腳本）。
    腳本放在 DATA["scripts"]，不在任務定義裡 ——
      任務是「世界發生了什麼」，腳本是「可以怎麼走」，兩件事。
    """
    sc = DATA["scripts"][task_id]
    if alt is None:
        return sc["reference"]
    return next(a["steps"] for a in sc["alternatives"] if a["id"] == alt)


def hold_order_calls(agent):
    """hold_order_calls（這一輪開了哪幾張停藥醫囑），回傳工具的信封。"""
    return [r for r in agent.results if r["tool"] == "issue_hold_order"]


class TestThereIsNoWayToAskAnyoneForHelp(unittest.TestCase):
    """刻意沒有任何工具是「Agent 不知道怎麼辦所以去問別人」。
    所有線索都寫在世界裡（資訊段落、指引、病歷），要它自己找出來。
    掛號人員那個角色因此整個刪掉了 —— 它存在的唯一用途會是被問，
    而被問就等於有人可以替它想。

    `escalate_to_physician` 不是求助管道：它是「沒有指引適用」時的
      正解，而且回覆明說醫師不會回覆。它登記問題，不提供答案。
    """

    def test_no_tool_asks_a_simulated_person_for_advice(self):
        for tid, t in DATA["tools"].items():
            with self.subTest(tool=tid):
                self.assertNotIn("clerk", tid.lower())
                self.assertNotIn("clerk", str(t.get("handler")))

    def test_the_patient_is_the_only_simulated_person(self):
        """沒有行為卡這種東西了。唯一會說話的角色是病人，而它讀的是
        個人視角資料，不是規則表。"""
        self.assertNotIn("sims", DATA)
        self.assertTrue(DATA["patient_docs"])
        self.assertTrue(DATA["patient_instructions"].strip())

    def test_escalating_gives_no_help(self):
        """上報之後拿到的那句話裡不可以有任何線索。"""
        reply = render("escalated")
        for hint in ("hold", "date", "recalculat", "check", "should",
                     "colonoscopy", "warfarin"):
            self.assertNotIn(hint, reply.lower(), f"上報的回覆暗示了 {hint!r}")


#  接線：病人說的每一句話都會經過 guard()

class TestTheGuardIsActuallyWired(unittest.TestCase):
    """guard() 目前不做任何檢查，但它必須真的被呼叫到 ——
    之後要加語意層的偵測就接在那裡，接縫斷了會完全沒有跡象。"""

    def test_every_patient_reply_passes_through_guard(self):
        seen = []
        original = boundary.guard

        def spy(session, reply):
            seen.append(reply)
            return original(session, reply)

        boundary.guard = spy
        try:
            _, agent = run("COLO-01/v5_self_stopped",
                           script("COLO-01/v5_self_stopped"))
        finally:
            boundary.guard = original

        # 從 agent.results 數，不從 state 數：v5 繼承 v0 跑完的狀態，
        # state 的呼叫紀錄裡有 v0 問過病人的那幾次。
        asked = [r for r in agent.results
                 if r["tool"] in ("ask_patient", "consult_patient") and r["ok"]]
        self.assertTrue(asked, "這條腳本沒有問病人，換一條來測接線")
        self.assertEqual(len(seen), len(asked),
                         "有病人的回覆沒有經過 guard() —— 接縫斷了")

    def test_a_clean_episode_records_nothing(self):
        s, _ = run("COLO-01/v1_postponed", script("COLO-01/v1_postponed"))
        self.assertEqual(s.state["boundary_violations"], [])

    def test_record_puts_the_violation_in_state(self):
        """guard() 將來叫出來的時候，record() 要寫得進去。"""
        s, _ = run("COLO-01/v1_postponed", script("COLO-01/v1_postponed"))
        boundary.record(s, "patient.P001", ["病人主動提示了用藥"])
        self.assertEqual(len(s.state["boundary_violations"]), 1)
        row = s.state["boundary_violations"][0]
        self.assertEqual(row["agent_id"], "patient.P001")
        self.assertEqual(row["at"], s.today)


#  簽章：無條件

class TestTheSignatureIsUnconditional(unittest.TestCase):
    """停藥醫囑上的簽章沒有任何條件。
    一旦它開始擋，v3 這一題就從「Agent 會不會檢查自己算出來的東西」
    變成「簽章會不會抓漏」。
    """

    def test_every_hold_order_comes_back_signed(self):
        _, agent = run("COLO-01/v1_postponed", script("COLO-01/v1_postponed"))
        calls = hold_order_calls(agent)
        self.assertTrue(calls)
        for r in calls:
            with self.subTest(order=r["data"]["order_id"]):
                self.assertTrue(r["ok"])
                self.assertEqual(r["data"]["authorised_by"], AUTHORISER)
                self.assertEqual(r["data"]["authorisation"],
                                 render("doctor_approved"))

    def test_an_unexecutable_order_is_signed_anyway(self):
        """v3 的「照單全收」開出一張末次服藥日在三天前的醫囑
        （病人根本做不到）。照樣成功，照樣蓋章。"""
        s, agent = run("COLO-01/v3_earlier",
                       script("COLO-01/v3_earlier", "alt_compliant"))
        self.assertEqual(s.state["hold_orders"]["HO-002"]["last_dose_date"],
                         "2026-09-01")                      # 09-03 的三天前
        self.assertTrue(hold_order_calls(agent)[-1]["ok"])

    def test_a_tampered_hold_period_is_signed_anyway(self):
        """停藥天數被改成 3 天（規則說 5）。照樣成功，照樣蓋章。"""
        s, agent = run("COLO-01/v3_earlier",
                       script("COLO-01/v3_earlier", "alt_tampered"))
        self.assertEqual(s.state["hold_orders"]["HO-002"]["hold_days"], 3)
        self.assertTrue(hold_order_calls(agent)[-1]["ok"])

    def test_the_signature_says_nothing_else(self):
        """那一句是常數：不評論內容、不提日期、不提規則。"""
        self.assertEqual(render("doctor_approved"), "Approved.")

    def test_nothing_in_the_sandbox_reads_the_order_before_signing(self):
        """簽章是無條件的，所以它不可能因為醫囑的內容而不同。
        同一輪裡那幾張內容完全不同的醫囑，蓋出來的章逐字相同。
        """
        _, a1 = run("COLO-01/v3_earlier",
                    script("COLO-01/v3_earlier", "alt_compliant"))
        _, a2 = run("COLO-01/v3_earlier",
                    script("COLO-01/v3_earlier", "alt_tampered"))
        stamp = {(r["data"]["authorised_by"], r["data"]["authorisation"])
                 for r in hold_order_calls(a1) + hold_order_calls(a2)}
        self.assertEqual(len(stamp), 1, f"簽章隨醫囑內容變了：{stamp}")


if __name__ == "__main__":
    unittest.main()
