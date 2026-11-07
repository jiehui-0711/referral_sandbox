"""主迴圈：六個案例都能用預設腳本從頭跑到尾。
這個檔不判對錯。它問的是「跑得動嗎」：

      腳本能不能走完
      事件有沒有在該觸發的那一天觸發
      時鐘有沒有在該前進的時候前進
      橫跨多個回合的腳本，日期有沒有落在對的地方
      同一條腳本跑兩次，結果有不有逐字相同
      案例之間有沒有互相污染

  「它做對了沒有」是驗證階段的事，而驗證方式還沒設計，
  所以這裡一個字都不提。

腳本的用途因此也變了：它是煙霧測試的素材，不是正解。
  別的走法（alternatives）在這裡只是「另外幾條會走到不同狀態
  的路」，用來確認那些路也走得通。
"""
from __future__ import annotations

import unittest

from rsandbox.agents.scripted import ScriptedAgent
from rsandbox.loaders import load_all
from rsandbox.runtime import initial, loop

DATA = load_all()
CASE = "COLO-01"
BASE = initial.clean_end(DATA, CASE)

# 不寫死任務清單：資料集會長到九十組，寫死的清單一定會跟資料脫節。
#   ALL      載入了的全部變體（已排序，所以順序是確定的）
#   PERTURBED  有擾動的那些（扣掉每一組的 v0）
ALL = sorted(DATA["tasks"])
PERTURBED = [t for t in ALL
             if (DATA["tasks"][t]["initial_state"] or {}).get("from") != "empty"]



def run(task_id: str, steps):
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


def alternatives_of(task_id: str):
    """alternatives_of（這個變體有哪幾條別的走法）。"""
    return (DATA["scripts"].get(task_id) or {}).get("alternatives") or []


#  事件觸發

class TestEvents(unittest.TestCase):

    def test_the_event_fires_on_its_own_date_and_nowhere_else(self):
        s, _ = run("COLO-01/v1_postponed", script("COLO-01/v1_postponed"))
        obs = s.state["observations"]
        with_event = [o for o in obs if o["events"]]
        self.assertEqual(len(with_event), 1)
        self.assertEqual(with_event[0]["at"], "2026-09-03")
        self.assertEqual(with_event[0]["events"], ["patient_call"])

    def test_the_event_text_reaches_the_agent_verbatim(self):
        """事件內容逐字轉述，不摘要 —— 摘要就是在替 Agent 決定什麼重要。"""
        task = DATA["tasks"]["COLO-01/v1_postponed"]
        content = task["event_script"][0]["content"].strip()
        s, agent = run("COLO-01/v1_postponed", script("COLO-01/v1_postponed"))
        self.assertIn(content, agent.observations[0])
        self.assertIn("2026-09-03", agent.observations[0])

    def test_the_observation_says_nothing_about_medication(self):
        """觀察每一回合都送到 Agent 眼前，所以它是最危險的洩漏管道。
        C1 的事件文字裡一個字都沒提到藥 —— 這一條把它釘住。

        觀察只給病人編號，不附病歷全文，所以整段都可以直接掃。
        """
        s, agent = run("COLO-01/v1_postponed", script("COLO-01/v1_postponed"))
        blob = "\n".join(agent.observations)
        for word in ("藥", "medication", "hold", "anticoagul", "重算", "recalculat",
                     "停藥", "注意", "記得"):
            self.assertNotIn(word, blob.lower() if word.isascii() else blob,
                             f"觀察裡出現了 {word!r}")

    def test_the_first_turn_carries_the_case_briefing(self):
        s, agent = run("COLO-01/v1_postponed", script("COLO-01/v1_postponed"))
        self.assertIn("P001", agent.observations[0])


#  煙霧測試：腳本走得完

class TestEveryScriptRuns(unittest.TestCase):
    """這一組就是「跑一輪看會不會出錯」。
    它只斷言三件事：
          - 整條腳本走完，沒有丟例外
          - 每一次呼叫都收到一個信封（不是 None、不是例外）
          - 預設腳本的每一步都被環境接受（沒有被關卡擋下來）
    ③ 只對預設腳本成立。變體腳本本來就可能踩到關卡
      （那是它們存在的理由），所以對它們只要求 ① 和 ②。
    """

    def test_every_default_script_runs_with_nothing_blocked(self):
        for tid in ALL:
            with self.subTest(task=tid):
                s, agent = run(tid, script(tid))
                self.assertEqual(agent.failures(), [],
                                 f"{tid} 的預設腳本有呼叫被擋下來："
                                 f"{agent.failures()}")
                self.assertTrue(s.state["tool_call_log"], f"{tid} 一次呼叫都沒有")

    def test_every_call_gets_an_envelope(self):
        """不管成功失敗，每一次呼叫都要留下一筆紀錄，而且形狀一致。"""
        for tid in ALL:
            rows = [(None, script(tid))] + [(f["id"], f["steps"])
                                          for f in alternatives_of(tid)]
            for fid, steps in rows:
                with self.subTest(task=tid, script=fid or "default"):
                    s, _ = run(tid, steps)
                    for c in s.state["tool_call_log"]:
                        self.assertEqual(
                            set(c) >= {"seq", "tool", "args", "ok", "reason",
                                       "data", "at"}, True, c)
                        self.assertIsInstance(c["ok"], bool)
                        if c["ok"]:
                            self.assertIsNone(c["reason"])
                        else:
                            self.assertIsNotNone(c["reason"])

    def test_every_variant_script_also_runs_to_the_end(self):
        """變體腳本可能被關卡擋下來，但不可以讓整輪爆掉。"""
        checked = 0
        for tid in ALL:
            for f in alternatives_of(tid):
                with self.subTest(task=tid, script=f["id"]):
                    s, _ = run(tid, f["steps"])
                    self.assertTrue(s.state["observations"])
                    checked += 1
        self.assertGreaterEqual(checked, 12, "變體腳本跑得太少")

    def test_no_script_makes_a_simulated_person_cross_the_line(self):
        """不變量④：模擬代理人不主動提示。
        它一越線，那一輪的資料就不能用。這跟「Agent 做得對不對」無關，
        所以就算刪掉了評分層，這一條還是要守。
        """
        for tid in ALL:
            rows = [(None, script(tid))] + [(f["id"], f["steps"])
                                          for f in alternatives_of(tid)]
            for fid, steps in rows:
                with self.subTest(task=tid, script=fid or "default"):
                    s, _ = run(tid, steps)
                    self.assertEqual(s.state["boundary_violations"] or [], [])


#  逐案：世界的性質

class TestC2ChangesOnlyWhatItTouches(unittest.TestCase):

    def test_the_default_script_changes_only_the_phone(self):
        """C2 是負控制：該做的只有改電話，別的一個欄位都不該動。"""
        s, _ = run("COLO-01/v2_contact_only", script("COLO-01/v2_contact_only"))
        self.assertEqual(s.state["appointments"]["APT-001"]["date"], "2026-09-10")
        self.assertEqual(s.state["hold_orders"]["HO-001"]["last_dose_date"],
                         "2026-09-05")
        self.assertEqual(len(s.state["prep_instructions"]), 1)
        self.assertEqual(s.records["P001"]["phone"], "0912-345-678")

    def test_the_over_reacting_variant_really_does_write_twice(self):
        """它多寄了一份說明單。環境不阻止 —— 那是刻意的。"""
        s, _ = run("COLO-01/v2_contact_only", script("COLO-01/v2_contact_only", "alt_over_reaction"))
        self.assertEqual(len(s.state["prep_instructions"]), 2)


class TestC3Feasibility(unittest.TestCase):

    def test_the_default_script_lands_on_a_day_that_exists(self):
        s, _ = run("COLO-01/v3_earlier", script("COLO-01/v3_earlier"))
        ap = s.state["appointments"]["APT-001"]
        self.assertEqual(ap["date"], "2026-09-08")     # 不是病人要求的 09-06
        slot = s.world["slots"][ap["slot_id"]]
        self.assertEqual(slot["date"], ap["date"])

    def test_an_order_whose_date_already_passed_is_still_accepted(self):
        """不變量①：末次服藥日是完全沒有約束的欄位。
        這條變體開出一張末次服藥日在三天前的醫囑 —— 物理上做不到。
        環境照樣收，而且醫師照樣簽。
        """
        s, agent = run("COLO-01/v3_earlier", script("COLO-01/v3_earlier", "alt_compliant"))
        self.assertEqual(s.state["hold_orders"]["HO-002"]["last_dose_date"],
                         "2026-09-01")
        issued = [c for c in s.state["tool_call_log"]
                  if c["tool"] == "issue_hold_order"]
        self.assertTrue(all(c["ok"] for c in issued), "開立停藥醫囑被擋下來了")

    def test_a_tampered_hold_period_is_accepted_too(self):
        """停藥天數被改成 3 天（規則說 5）。環境不比對規則，照樣收。"""
        s, _ = run("COLO-01/v3_earlier", script("COLO-01/v3_earlier", "alt_tampered"))
        self.assertEqual(s.state["hold_orders"]["HO-002"]["hold_days"], 3)


class TestC4TheWorldIsLegalButInconsistent(unittest.TestCase):

    def test_the_cancelled_slot_disappears_from_the_listing(self):
        s, agent = run("COLO-01/v4_slot_cancelled", script("COLO-01/v4_slot_cancelled"))
        first = next(r for r in agent.results if r["tool"] == "query_slots")
        dates = [row["date"] for row in first["data"]["slots"]]
        self.assertNotIn("2026-09-10", dates)
        self.assertIn("2026-09-15", dates)

    def test_a_booked_appointment_may_point_at_a_cancelled_slot(self):
        """C4 的初始狀態是合法但矛盾的：一張 booked 的預約指向一個
        已取消的時段。沒有任何約束擋它 —— 這就是「每一格都合法」的另一例。"""
        task = DATA["tasks"]["COLO-01/v4_slot_cancelled"]
        s = initial.build(DATA, task, BASE)
        self.assertEqual(
            s.world["slots"]["H1-GI-COL-20260910-AM"]["status"], "cancelled")
        self.assertEqual(s.state["appointments"]["APT-001"]["status"], "booked")
        self.assertEqual(s.state["appointments"]["APT-001"]["slot_id"],
                         "H1-GI-COL-20260910-AM")

    def test_a_script_can_span_three_turns(self):
        """這條腳本橫跨三個回合，所以它同時也在測主迴圈：
        如果迴圈一看到空回合就結束，這條根本跑不出來。"""
        s, agent = run("COLO-01/v4_slot_cancelled", script("COLO-01/v4_slot_cancelled", "alt_late"))
        self.assertEqual(agent.failures(), [])
        turns = [o["at"] for o in s.state["observations"]]
        self.assertEqual(turns, ["2026-09-04", "2026-09-05", "2026-09-06"])
        self.assertEqual(s.state["prep_instructions"]["PREP-002"]["delivered_at"],
                         "2026-09-06")


class TestTheVariantPatchesThePatientDocument(unittest.TestCase):
    """v5 跟前四題本質不同：預約沒變、規則沒變、醫囑沒變。
    變的是病人的個人資料 —— 他自己先停了藥。"""

    def test_the_booking_and_the_order_are_untouched(self):
        s, _ = run("COLO-01/v5_self_stopped", script("COLO-01/v5_self_stopped"))
        self.assertEqual(s.state["appointments"]["APT-001"]["date"], "2026-09-10")
        self.assertEqual(s.state["hold_orders"]["HO-001"]["last_dose_date"],
                         "2026-09-05")

    def test_the_patch_changed_what_the_patient_knows(self):
        v0 = initial.build(DATA, DATA["tasks"]["COLO-01/v0_clean"])
        v5 = initial.build(DATA, DATA["tasks"]["COLO-01/v5_self_stopped"])
        key = "how_i_take_them"
        self.assertNotEqual(v0.patient_doc["knows"][key],
                            v5.patient_doc["knows"][key])
        self.assertIn("28 August 2026", v5.patient_doc["knows"][key])
        self.assertNotIn("28 August", v0.patient_doc["knows"][key])

    def test_the_patch_does_not_leak_into_the_other_variants(self):
        """每個 episode 一份病人資料的副本，不然改動會殘留到下一題。"""
        initial.build(DATA, DATA["tasks"]["COLO-01/v5_self_stopped"])
        again = initial.build(DATA, DATA["tasks"]["COLO-01/v0_clean"])
        self.assertNotIn("28 August", again.patient_doc["knows"]["how_i_take_them"])


class TestPatientSeesOnlyWhatHasHappened(unittest.TestCase):
    """病人只看得到日期 ≤ 今天的事件 —— 不然他會知道還沒發生的事。"""

    def test_a_future_event_is_not_visible(self):
        from rsandbox.sim.patient import visible_log
        doc = {"log": [{"date": "2026-09-03", "text": "a"},
                       {"date": "2026-09-20", "text": "b"}]}
        self.assertEqual([e["text"] for e in visible_log(doc, "2026-09-03")], ["a"])

    def test_the_patient_call_reaches_the_patient_document(self):
        """同一份事件宣告供兩邊：Agent 的觀察，與病人的個人資料。"""
        s, agent = run("COLO-01/v1_postponed", script("COLO-01/v1_postponed"))
        log = s.patient_doc["log"]
        self.assertTrue(log, "病人打了電話，但他自己的資料裡沒有這件事")
        self.assertEqual(log[0]["date"], "2026-09-03")
        self.assertIn("move it to the 17th", log[0]["text"])
        # 轉診 Agent 也收到了同一件事
        self.assertIn("move it to the 17th", agent.observations[0])

    def test_a_system_event_does_not_reach_the_patient(self):
        """醫院取消時段、環境發的提醒，病人都不知道。"""
        s, _ = run("COLO-01/v4_slot_cancelled", script("COLO-01/v4_slot_cancelled"))
        self.assertEqual(s.patient_doc["log"], [])


#  確定性與不污染

class TestDoingNothingIsAllowed(unittest.TestCase):
    """什麼都不做也要跑得完。
    以前這裡有一組測試叫「什麼都不做絕對不能及格」—— 那是評分的事，
      已經刪掉。現在這裡只確認它跑得動而且什麼都沒改，
      因為那是之後任何驗證方式都要用到的對照點。
    """

    NOTHING = [{"tool": "end_turn", "args": {"note": "Noted."}}]

    def test_doing_nothing_runs_and_changes_nothing(self):
        """比較的基準是「這一題開始時的狀態」，不是空的。
        C1～C5 繼承基準流程跑完的狀態，所以它們的變更紀錄一開始就有
        七十幾列（那是 baseline 留下的）。這一條要問的是
        這一輪有沒有再加上任何一列。
        """
        for tid in ALL:
            with self.subTest(task=tid):
                task = DATA["tasks"][tid]
                before = len(initial.build(DATA, task, BASE).state["change_log"])
                s, agent = run(tid, self.NOTHING)
                self.assertEqual(agent.failures(), [])
                added = s.state["change_log"][before:]
                self.assertEqual([r for r in added if r["tool"] != "end_turn"], [],
                                 f"{tid}：什麼都沒做卻改了資料")


class TestDeterminism(unittest.TestCase):

    def test_two_runs_of_every_case_are_byte_identical(self):
        for tid in ALL:
            steps = script(tid)
            a, _ = run(tid, steps)
            b, _ = run(tid, steps)
            with self.subTest(task=tid):
                self.assertEqual(a.state["tool_call_log"],
                                 b.state["tool_call_log"])
                self.assertEqual(a.state["observations"], b.state["observations"])
                self.assertEqual(a.state["change_log"], b.state["change_log"])
                self.assertEqual(a.snapshot(), b.snapshot())

    def test_the_cases_do_not_contaminate_each_other(self):
        """病歷是「載入時靜態、執行中可寫」的混合型，最容易踩的坑。"""
        c1, _ = run("COLO-01/v1_postponed", script("COLO-01/v1_postponed"))
        c2, _ = run("COLO-01/v2_contact_only", script("COLO-01/v2_contact_only"))
        self.assertEqual(c1.records["P001"]["phone"], "0912-345-678")
        self.assertEqual(c2.records["P001"]["phone"], "0912-345-678")
        self.assertEqual(DATA["records"]["P001"]["phone"], "0911-222-333")
        # 種子與 baseline 快照都沒被污染
        self.assertEqual(BASE["records"]["P001"]["phone"], "0911-222-333")

    def test_the_loop_stops_instead_of_idling_to_the_end_date(self):
        """如果只看結束日期，C1 會從 09-03 空轉到 09-30，
        而每個空回合都給 Agent 一次「再去改點什麼」的機會。"""
        s, _ = run("COLO-01/v1_postponed", script("COLO-01/v1_postponed"))
        self.assertEqual(s.state["turns"], 1)
        self.assertLessEqual(len(s.state["observations"]), 2)


if __name__ == "__main__":
    unittest.main()
