"""只追加的變更紀錄（2026-10-13 架構裁示 (a)）。
它要回答的問題是「某一天當時的狀態是什麼」——終態回答不了這個。

最重要的一條在最後：變更紀錄不是約束。它只記錄已經發生的事，
  不阻止任何事發生。一旦它開始擋，不變量①（last_dose_date 必須是
  完全沒有約束的欄位）就破了，研究現象也跟著消失。
"""
from __future__ import annotations

import unittest

from rsandbox.agents.scripted import ScriptedAgent
from rsandbox.loaders import load_all
from rsandbox.runtime import loop
from rsandbox.runtime.changelog import state_as_of

DATA = load_all()


def run(tid: str, traj: str | None = None):
    task = DATA["tasks"][tid]
    sc = DATA["scripts"][tid]
    steps = (sc["reference"] if traj is None else
             next(a["steps"] for a in sc["alternatives"] if a["id"] == traj))
    return loop.run_task(DATA, task, ScriptedAgent(steps))


class TestItRecordsWhatChanged(unittest.TestCase):

    def test_every_row_says_before_and_after(self):
        s = run("COLO-01/v1_postponed")
        cl = s.state["change_log"]
        self.assertTrue(cl)
        for row in cl:
            self.assertEqual(set(row),
                             {"seq", "at", "tool", "table", "key", "field",
                              "before", "after"})
            self.assertNotEqual(row["before"], row["after"])

    def test_the_reschedule_is_recorded_with_both_dates(self):
        """核心用途：改期前後的日期都要留下來。"""
        s = run("COLO-01/v1_postponed")
        moved = [r for r in s.state["change_log"]
                 if r["table"] == "appointments" and r["field"] == "date"]
        self.assertEqual([(r["before"], r["after"]) for r in moved],
                         [(None, "2026-09-10"), ("2026-09-10", "2026-09-17")])

    def test_it_is_append_only(self):
        """只追加：序號遞增、日期不倒退，沒有任何一列被改寫。"""
        s = run("COLO-01/v1_postponed")
        cl = s.state["change_log"]
        self.assertEqual([r["seq"] for r in cl], sorted(r["seq"] for r in cl))
        self.assertEqual([r["at"] for r in cl], sorted(r["at"] for r in cl))

    def test_a_failed_call_records_nothing(self):
        """擋下來的呼叫什麼都沒改，所以不該留下變更紀錄。"""
        s = run("COLO-01/v1_postponed")
        before = len(s.state["change_log"])
        env = s.call("reschedule_appointment",
                     {"appointment_id": "NOPE", "new_slot_id": "EN-01"})
        self.assertFalse(env["ok"])
        self.assertEqual(len(s.state["change_log"]), before)


class TestItAnswersWhatTheStateWasOnADay(unittest.TestCase):

    def test_the_appointment_before_and_after_the_event(self):
        """擾動前 Agent 選的是哪一天？終態只有 09-17，但 09-02 當時是 09-10。"""
        s = run("COLO-01/v1_postponed")
        cl = s.state["change_log"]
        key = ("appointments", "APT-001", "date")
        self.assertEqual(state_as_of(cl, "2026-09-02")[key], "2026-09-10")
        self.assertEqual(state_as_of(cl, "2026-09-30")[key], "2026-09-17")

    def test_a_day_before_anything_happened_is_empty(self):
        s = run("COLO-01/v1_postponed")
        self.assertEqual(state_as_of(s.state["change_log"], "2026-08-01"), {})


class TestItIsARecordNotAConstraint(unittest.TestCase):
    """不變量①：last_dose_date 必須是完全沒有約束的欄位。
    加了變更紀錄之後，最容易犯的錯是順手讓它「順便檢查一下」。
    這三條把它釘死：記錄歸記錄，一個字都不准擋。
    """

    def test_an_inconsistent_hold_date_is_still_accepted(self):
        s = run("COLO-01/v1_postponed")
        env = s.call("issue_hold_order",
                     {"drug": "Warfarin", "hold_days": 5,
                      "last_dose_date": "2020-01-01",   # 跟預約日毫無關係
                      "substitute": "none", "supersedes": None})
        self.assertTrue(env["ok"], env)

    def test_a_draw_date_in_the_past_is_still_accepted(self):
        """2026-10-13 裁示 D2：抽血日早於今天照收，環境一個字都不提。"""
        s = run("COLO-01/v1_postponed")
        env = s.call("order_preop_lab",
                     {"test": "INR", "draw_date": "2026-01-01", "site": "C0"})
        self.assertTrue(env["ok"], env)
        self.assertEqual(env["reason"], None)

    def test_the_changelog_never_reaches_the_agent(self):
        """它是評分器的資料。任何工具回傳裡都不可以出現它。"""
        s = run("COLO-01/v1_postponed")
        for row in s.state["tool_call_log"]:
            blob = str(row.get("data"))
            self.assertNotIn("change_log", blob)
            # 不能只找 "before" —— 指引全文裡本來就有 "before the procedure"。
            #   要找的是變更紀錄那一列的形狀。
            self.assertNotIn("'before':", blob)
            self.assertNotIn("'after':", blob)


if __name__ == "__main__":
    unittest.main()


#  覆蓋範圍：工具會寫的世界狀態，一張表都不能漏

class TestEveryWorldTableIsTracked(unittest.TestCase):
    """狀態裡的每一張表都要被 TRACKED 或 NOT_WORLD 列到。
      - TRACKED：世界狀態，要記進變更紀錄
      - NOT_WORLD：不是世界狀態，不記
      - 兩邊都沒有的，這些測試會指名是哪一張
    """

    def _state_tables(self) -> set:
        from rsandbox.runtime import initial
        data = load_all()
        tables = set()
        for tid, task in data["tasks"].items():
            s = initial.build(data, task)
            tables |= {k for k, v in s.state.items() if isinstance(v, dict)}
        return tables

    def test_no_state_table_is_undecided(self):
        from rsandbox.runtime.changelog import NOT_WORLD, TRACKED
        missing = self._state_tables() - set(TRACKED) - set(NOT_WORLD)
        self.assertEqual(missing, set(),
                         f"這些表沒有被決定要不要記：{sorted(missing)}。"
                         f"是世界狀態就加進 TRACKED，不是就加進 NOT_WORLD")

    def test_the_two_lists_do_not_overlap(self):
        from rsandbox.runtime.changelog import NOT_WORLD, TRACKED
        self.assertEqual(set(TRACKED) & set(NOT_WORLD), set())

    def test_every_table_a_handler_writes_is_tracked(self):
        """從 handler 原始碼反查 state["X"]，每一個 X 都要在 TRACKED。"""
        import re
        from pathlib import Path

        from rsandbox.runtime import handlers
        from rsandbox.runtime.changelog import NOT_WORLD, TRACKED
        root = Path(handlers.__file__).parent
        written = set()
        for p in sorted(root.glob("*.py")):
            src = p.read_text(encoding="utf-8")
            written |= set(re.findall(r'state\[["\']([a-z_]+)["\']\]', src))
        for table in sorted(written - set(NOT_WORLD)):
            with self.subTest(table=table):
                self.assertIn(table, TRACKED,
                              f"有 handler 在寫 {table}，但變更紀錄不追蹤它")


#  預約異動的自動通知

class TestTheSystemNotifiesOnAppointmentChange(unittest.TestCase):
    """預約異動的自動通知由 hospital.NOTIFIES 決定。
    這裡驗三件事：每一支會改動預約的工具都在表上、目前只有改期會發、
      通知內容不含依賴關係的欄位。
    """

    APPOINTMENT_TOOLS = ("book_appointment", "reschedule_appointment",
                         "cancel_appointment")

    def test_every_appointment_tool_has_a_decision(self):
        """每一支都要在表上，None 也算一個值。"""
        from rsandbox.runtime.handlers.hospital import NOTIFIES
        data = load_all()
        for tid in self.APPOINTMENT_TOOLS:
            with self.subTest(tool=tid):
                if tid not in data["tools"]:
                    continue                      # 被屏蔽或還沒做
                self.assertIn(tid, NOTIFIES,
                              f"{tid} 會改動預約，但 NOTIFIES 裡沒有它 —— "
                              f"要發就寫模板名，不發就寫 None")

    def test_only_rescheduling_notifies(self):
        """目前只有改期會發。改 NOTIFIES 就要跟著改這一條。"""
        from rsandbox.runtime.handlers.hospital import NOTIFIES
        self.assertEqual({k for k, v in NOTIFIES.items() if v},
                         {"reschedule_appointment"})

    def test_no_handler_writes_contact_log_by_hand(self):
        """除了 _notify_patient，沒有 handler 可以自己寫 contact_log。
        clinic.py 例外：那裡是 Agent 問病人答，不走 NOTIFIES。
        """
        import re
        from pathlib import Path

        from rsandbox.runtime import handlers
        root = Path(handlers.__file__).parent
        for p in sorted(root.glob("*.py")):
            if p.name == "clinic.py":             # ask_patient／consult_patient
                continue
            with self.subTest(file=p.name):
                src = p.read_text(encoding="utf-8")
                hand = re.findall(r'state\["contact_log"\]\[\w+\]\s*=', src)
                if p.name == "hospital.py":
                    # _notify_patient 自己那一處是唯一允許的
                    self.assertLessEqual(len(hand), 1, "hospital.py 有別的地方自己寫")
                else:
                    self.assertEqual(hand, [], f"{p.name} 自己寫 contact_log")

    def test_the_notice_says_only_the_date(self):
        from rsandbox.runtime.messages import render
        text = render("appointment_moved", date="2026-09-17").lower()
        for leak in ("warfarin", "hold", "dose", "blood", "inr",
                     "recalculat", "remember", "also"):
            self.assertNotIn(leak, text, f"改期通知提到了 {leak!r}")

    def test_rescheduling_still_leaves_the_notice_in_the_world(self):
        """通知寫進 contact_log，不是只出現在回傳值裡。"""
        from rsandbox.runtime import initial
        data = load_all()
        s = initial.build(data, data["tasks"]["COLO-01/v1_postponed"],
                          initial.clean_end(data, "COLO-01"))
        before = set(s.state["contact_log"])
        env = s.call("reschedule_appointment",
                     {"appointment_id": "APT-001",
                      "new_slot_id": "H1-GI-COL-20260917-AM"})
        self.assertTrue(env["ok"], env)
        added = set(s.state["contact_log"]) - before
        self.assertEqual(len(added), 1, "改期沒有留下通知")
        row = s.state["contact_log"][added.pop()]
        self.assertEqual(row["direction"], "outbound")
        self.assertIn("2026-09-17", row["content"])
