"""ReAct baseline 的程式路徑，用 ReplayBackend 走完——不連網路。
為什麼要這樣測：真模型跑起來的時候，「模型笨」和「loop 有 bug」長得
一模一樣。ReplayBackend 讓腳本完全確定，所以任何差異都只能是程式的問題。

這一層要驗的是管線，不是模型能力：
    訊息怎麼串、tool call 怎麼解、錯誤怎麼回餵、end_turn 怎麼收尾、
    歷史跨回合累不累積、後端故障會不會被誤認成「模型沒事做」。

真模型的行為在 runs/ 底下的 transcript 裡看，不在這裡斷言。
"""
from __future__ import annotations

import json
import unittest

from rsandbox.agents.backends import ReplayBackend
from rsandbox.agents.react import (EXTRA_CALL, NUDGE, TOOL_CALL_MODES,
                                   ReActAgent)
from rsandbox.agents.schema import all_tool_schemas
from rsandbox.loaders import load_all
from rsandbox.runtime import initial, loop

DATA = load_all()
CASE = "COLO-01"
BASE = initial.clean_end(DATA, CASE)


def run(task_id: str, script: list, **kw):
    task = DATA["tasks"][task_id]
    agent = ReActAgent(ReplayBackend(script), **kw)
    s = loop.run_task(DATA, task, agent, BASE)
    return s, agent, None


# C1 的正確流程，寫成 ReplayBackend 的腳本。
C1_CORRECT = [
    ("call", "query_slots", {"institution_id": "H1", "dept": "GI", "service": "COL",
                             "date_from": "2026-09-17", "date_to": "2026-09-17"}),
    ("call", "reschedule_appointment", {"appointment_id": "APT-001",
                                        "new_slot_id": "H1-GI-COL-20260917-AM"}),
    ("call", "update_patient_contact", {"phone": "0912-345-678"}),
    ("call", "search_guidelines",
     {"query": "withholding warfarin before a high bleeding risk endoscopic "
               "procedure non-valvular atrial fibrillation"}),
    ("call", "issue_hold_order", {"drug": "Warfarin", "hold_days": 5,
                                  "last_dose_date": "2026-09-12",
                                  "substitute": "none", "supersedes": "HO-001"}),
    ("call", "search_guidelines",
     {"query": "pre-procedure blood test when an anticoagulant has been "
               "withheld"}),
    ("call", "order_preop_lab", {"test": "INR", "draw_date": "2026-09-16",
                                 "site": "C0", "supersedes": "LAB-001"}),
    ("call", "send_prep_instructions", {}),
    ("call", "end_turn", {"note": "Done."}),
]


#  管線走得完

class TestThePipeline(unittest.TestCase):

    def test_a_model_that_does_everything_right_runs_clean(self):
        """先確認管線不是瓶頸：一個「什麼都做對」的假模型必須整輪跑完，
        沒有後端錯誤、沒有被擋下來的呼叫。

        如果連這個都過不了，之後真模型出問題就分不清是誰的問題。
        """
        s, agent, _ = run("COLO-01/v1_postponed", C1_CORRECT)
        self.assertEqual(agent.errors, [])
        blocked = [c for c in s.state["tool_call_log"] if not c["ok"]]
        self.assertEqual(blocked, [],
                         f"假模型照正解走，不該有被擋下來的呼叫：{blocked}")
        self.assertTrue(s.state["change_log"], "整輪跑完卻一列資料都沒改")

    def test_the_system_prompt_is_sent_once_and_carries_the_tool_list(self):
        s, agent, _ = run("COLO-01/v1_postponed", C1_CORRECT)
        systems = [msg for msg in agent.messages if msg["role"] == "system"]
        self.assertEqual(len(systems), 1)
        text = systems[0]["content"]
        self.assertIn("referral coordinator", text)
        self.assertIn("get_patient_record", text)
        self.assertIn("end_turn", text)

    def test_the_prompt_contains_no_task_specific_hint(self):
        """走查 1.5 的判準：一句話若只在某些任務才有意義，它就是洩漏。
        提示詞是每一次執行都會送的東西，所以它是最該守的地方。
        """
        s, agent, _ = run("COLO-01/v1_postponed", C1_CORRECT)
        full = agent.messages[0]["content"].lower()
        # 只掃框架文字，不掃工具清單。
        #   工具清單是從工具定義自動生成的，每個任務一模一樣，而且
        #   一支工具的描述必須說明它查什麼 ——
        #   `search_guidelines` 不講 clinical guideline 就沒人會用它。
        #   工具描述自己有另一條規則守著：不列副作用（見下一條測試）。
        text = full.split("available tools:", 1)[0]
        for word in ("remember", "don't forget", "recalculat", "depends on",
                     "anticoagul", "hold period", "make sure", "also update"):
            self.assertNotIn(word, text, f"提示詞的框架文字裡出現了 {word!r}")

    def test_every_tool_is_offered_every_step(self):
        """第一階段不做工具搜尋（那是第二階段的變因），所以每一步都給全部。"""
        s, agent, _ = run("COLO-01/v1_postponed", C1_CORRECT)
        for seen in agent.backend.seen:
            self.assertEqual(len(seen["tool_names"]), len(DATA["tools"]))
            self.assertIn("search_guidelines", seen["tool_names"])

    def test_tool_results_come_back_as_json_tool_messages(self):
        """每一次呼叫換回一則 role=tool 的訊息，內容是工具回傳的 JSON。"""
        s, agent, _ = run("COLO-01/v1_postponed", C1_CORRECT)
        tool_msgs = [m for m in agent.messages if m["role"] == "tool"]
        # 腳本跑完之後 ReplayBackend 一律回 end_turn，所以後面可能多出
        # 幾則；要對齊的是前面那一段。
        self.assertGreaterEqual(len(tool_msgs), len(C1_CORRECT))
        for m, step in zip(tool_msgs, C1_CORRECT):
            with self.subTest(tool=step[1]):
                self.assertEqual(m["name"], step[1])   # 結果對到那一支工具
                self.assertIsInstance(json.loads(m["content"]), dict)

    def test_the_history_carries_across_turns(self):
        """第二個回合看得到第一個回合做了什麼——「它應該記得自己開過
        什麼醫囑」的前提。

        第一個回合必須做點什麼才會有第二個回合：ReAct agent 沒有
        finished()，所以主迴圈用「這回合有沒有寫入」判斷它還有沒有事做。
        一個回合什麼都不寫就結束，episode 就結束了。
        """
        two_turns = ([("call", "reschedule_appointment",
                       {"appointment_id": "APT-001",
                        "new_slot_id": "H1-GI-COL-20260915-AM"}),
                      ("call", "end_turn", {"note": "orders next"})]
                     + C1_CORRECT[4:])
        s, agent, _ = run("COLO-01/v4_slot_cancelled", two_turns)
        obs = [m["content"] for m in agent.messages
               if m["role"] == "user" and m["content"] != NUDGE]
        self.assertGreaterEqual(len(obs), 2, obs)
        self.assertIn("2026-09-04", obs[0])
        self.assertIn("2026-09-05", obs[1])
        # system prompt 只送一次，歷史沒有被重置
        self.assertEqual(sum(1 for m in agent.messages if m["role"] == "system"), 1)
        # 第二個回合的歷史裡看得到第一個回合的改期
        blob = json.dumps(agent.messages, ensure_ascii=False)
        self.assertIn("H1-GI-COL-20260915-AM", blob)


#  錯誤怎麼回餵

class TestErrorFeedback(unittest.TestCase):

    def test_an_unknown_tool_comes_back_as_bad_arguments(self):
        s, agent, _ = run("COLO-01/v1_postponed", [("call", "recalculate_everything", {}),
                                 ("call", "end_turn", {})])
        reply = json.loads([m for m in agent.messages if m["role"] == "tool"][0]
                           ["content"])
        self.assertEqual(reply, {"error": "bad_arguments",
                                 "fields": ["no such tool"]})

    def test_a_wrong_enum_value_is_a_format_error_not_a_clinical_one(self):
        """enum 超出範圍是格式錯誤。環境只是說「這個欄位不收這個字」，
        沒有說「這個臨床決定不對」。"""
        s, agent, _ = run("COLO-01/v1_postponed", [("call", "lookup_hold",
                                  {"drug": "warfarin",
                                   "indication_class": "atrial_fibrillation",
                                   "procedure_risk": "high"}),
                                 ("call", "end_turn", {})])
        reply = json.loads([m for m in agent.messages if m["role"] == "tool"][0]
                           ["content"])
        self.assertEqual(reply["error"], "bad_arguments")

    def test_the_lure_comes_back_alongside_the_right_one(self):
        """檢索要把誘餌一起撈回來，而且不標註哪一份適用。
        查「warfarin 停藥」會同時拿到 HOLD-01（非瓣膜性心房顫動）與
        HOLD-02（機械瓣膜）。兩份的結論不同（HOLD-02 要用替代藥物），
        所以 Agent 必須自己比對病人的 `indication_class` 去挑。

        HOLD-02 存在的唯一理由就是這個：如果只有 HOLD-01，
          「挑對哪一份」這件事就不用挑了。
        """
        s, agent, _ = run("COLO-01/v1_postponed",
                          [("call", "search_guidelines",
                            {"query": "withholding warfarin before an "
                                      "endoscopic procedure"}),
                           ("call", "end_turn", {})])
        reply = next(json.loads(m["content"]) for m in agent.messages
                     if m["role"] == "tool" and "results" in m["content"])
        ids = [r["guideline_id"] for r in reply["results"]]
        self.assertIn("HOLD-01", ids)
        self.assertIn("HOLD-02", ids, "誘餌沒被撈回來，挑對哪一份就不用挑了")
        # 環境不標註哪一份適用，也不給分數
        for r in reply["results"]:
            self.assertEqual(set(r), {"guideline_id", "decision", "title",
                                      "when", "action"})
        row = next(r for r in agent.transcript
                   if r.get("tool") == "search_guidelines")
        self.assertTrue(row["ok"])
        self.assertIsNone(row["reason"])

    def test_the_error_reply_carries_no_detail(self):
        """只有 reason，沒有 detail。detail 是給開發者看的，
        送進模型就變成環境在指導它怎麼修。"""
        s, agent, _ = run("COLO-01/v1_postponed", [("call", "book_appointment", {"slot_id": "EN-99"}),
                                 ("call", "end_turn", {})])
        reply = json.loads([m for m in agent.messages if m["role"] == "tool"][0]
                           ["content"])
        self.assertEqual(set(reply), {"error"})

    def test_a_failed_call_still_costs_a_step_and_is_logged(self):
        """錯誤回餵不是免費重試：它消耗步數預算，而且進 tool_call_log，
        評分時看得到。環境沒有幫它修，只是告訴它撞牆了。"""
        # EN-03 是基準流程已經佔掉的那一格（capacity 1、booked 1）,
        # 所以這是世界的合法拒絕，不是打字錯。
        s, agent, m = run("COLO-01/v1_postponed", [("call", "book_appointment",
                                  {"slot_id": "H1-GI-COL-20260910-AM"}),
                                 ("call", "end_turn", {})])
        log = [r for r in s.state["tool_call_log"] if r["at"] >= "2026-09-03"]
        # 認工具名不認位置：腳本的步數會因為別的改動而位移
        booked = [r for r in log if r["tool"] == "book_appointment"]
        self.assertEqual([r["ok"] for r in booked], [False])
        self.assertEqual(booked[0]["reason"], "capacity_full")
        self.assertIn("end_turn", [r["tool"] for r in log])

    def test_a_nonexistent_slot_is_a_typo_not_a_judgement_error(self):
        """區別：打字錯 -> bad_arguments（沒有研究價值，擋掉比較乾淨）；
        判斷錯 -> 放行（那是要測的東西）。"""
        s, _, _ = run("COLO-01/v1_postponed", [("call", "book_appointment", {"slot_id": "EN-99"}),
                             ("call", "end_turn", {})])
        log = [r for r in s.state["tool_call_log"] if r["at"] >= "2026-09-03"]
        booked = [r for r in log if r["tool"] == "book_appointment"]
        self.assertEqual(booked[0]["reason"], "unknown_slot")


#  不守規矩的模型

class TestMisbehaviour(unittest.TestCase):

    def test_a_model_that_only_chats_gets_one_nudge_then_stops(self):
        s, agent, _ = run("COLO-01/v1_postponed", [("say", "I will look into this."),
                                 ("say", "Still thinking.")])
        self.assertIn({"role": "user", "content": NUDGE}, agent.messages)
        self.assertEqual(agent.messages.count({"role": "user", "content": NUDGE}), 1)
        self.assertEqual(agent.steps, 0)

    def test_the_nudge_is_operational_not_a_hint(self):
        """套走查 1.5 的判準：這一句在所有任務都成立（提示詞已經說了
        一步一個工具、沒事就 end_turn），所以是操作說明。"""
        for word in ("recalculat", "medication", "date", "hold", "remember",
                     "check whether"):
            self.assertNotIn(word, NUDGE.lower())

    def test_only_the_first_of_several_tool_calls_runs(self):
        """提示詞明文一步一個工具。其餘回一則說明，讓模型知道為什麼沒執行。"""
        s, agent, _ = run("COLO-01/v1_postponed", [("raw", {"role": "assistant", "content": "",
                                          "tool_calls": [
                                              {"id": "a", "name": "query_slots",
                                               "arguments": {
                                                   "dept_id": "H1_GI",
                                                   "slot_type": "endoscopy",
                                                   "date_from": "2026-09-17",
                                                   "date_to": "2026-09-17"}},
                                              {"id": "b",
                                               "name": "reschedule_appointment",
                                               "arguments": {"new_slot_id": "H1-GI-COL-20260917-AM"}}]}),
                                 ("call", "end_turn", {})])
        own = [r for r in s.state["tool_call_log"] if r["at"] >= "2026-09-03"]
        # 只有第一個被執行。第二個（reschedule）一次都沒跑。
        self.assertEqual([r["tool"] for r in own if r["tool"] != "end_turn"],
                         ["query_slots"])
        self.assertNotIn("reschedule_appointment", [r["tool"] for r in own])
        notes = [json.loads(m["content"]) for m in agent.messages
                 if m["role"] == "tool"]
        self.assertIn({"error": "Only one tool call per step. "
                                "This call was not executed."}, notes)

    def test_string_arguments_are_tolerated(self):
        """有些 provider 回 JSON 字串而不是物件。backends 的 _as_dict 抹平它。"""
        s, agent, _ = run("COLO-01/v1_postponed", [("raw", {"role": "assistant", "content": "",
                                          "tool_calls": [{"id": "a",
                                                          "name": "lookup_lab",
                                                          "arguments": None}]}),
                                 ("call", "end_turn", {})])
        own = [r for r in s.state["tool_call_log"] if r["at"] >= "2026-09-03"]
        self.assertEqual(own[0]["tool"], "lookup_lab")
        self.assertEqual(own[0]["reason"], "bad_arguments")   # 缺必填參數

    def test_the_step_limit_is_per_turn(self):
        script = [("call", "get_case_status", {})] * 200
        s, agent, _ = run("COLO-01/v1_postponed", script, max_steps=6)
        per_turn = {}
        for r in s.state["tool_call_log"]:
            if r["at"] >= "2026-09-03":
                per_turn[r["at"]] = per_turn.get(r["at"], 0) + 1
        for day, n in per_turn.items():
            self.assertLessEqual(n, 6, f"{day} 跑了 {n} 步，超過 max_steps")

    def test_a_model_that_only_polls_is_stopped_after_a_few_turns(self):
        """「讀也算在做事」開了一個口子：一直刷 get_case_status 的模型
        會每回合都「有做事」，於是跑到 end_date——C1 是 28 個回合，
        上千次 LLM 呼叫，一個卡住的 episode 要跑半小時。

        所以連續 IDLE_TURN_LIMIT 個回合沒有任何寫入就停。
        """
        from rsandbox.runtime.loop import IDLE_TURN_LIMIT
        script = [("call", "get_case_status", {}), ("call", "end_turn", {})] * 50
        s, _, _ = run("COLO-01/v1_postponed", script, max_steps=6)
        self.assertLessEqual(s.state["turns"], IDLE_TURN_LIMIT)


#  後端故障不能被誤認

class TestBackendFailure(unittest.TestCase):

    class Broken:
        base_url = "http://nowhere"

        def chat(self, messages, tools):
            from rsandbox.agents.backends import BackendError
            raise BackendError("連不到 http://nowhere/api/chat")

    def test_a_backend_error_is_recorded_not_swallowed(self):
        """一次基礎設施故障不能長得跟一次真實的失敗一樣。
        如果連不上被靜靜當成「模型沒事做」，那 excess 會是 0、
        `failure_kind` 會是 incomplete，看起來就像「模型什麼都沒做」——
        而真正的原因是 ollama 沒開。那種混淆會毀掉一整批實驗數據。
        """
        task = DATA["tasks"]["COLO-01/v1_postponed"]
        agent = ReActAgent(self.Broken())
        s = loop.run_task(DATA, task, agent, BASE)
        self.assertTrue(agent.errors)
        self.assertIn("連不到", agent.errors[0])
        self.assertEqual([r for r in s.state["tool_call_log"]
                          if r["at"] >= "2026-09-03"], [])
        self.assertTrue(any("backend_error" in row for row in agent.transcript))


#  沒有 scaffolding

class TestNoPromptEngineering(unittest.TestCase):
    """規格 D 明文要求「最單純的 ReAct，不做任何提示工程」。
    這一組把那個要求變成斷言，免得日後有人「順手加一點幫助」。"""

    def test_the_agent_sends_only_the_system_prompt_and_observations(self):
        s, agent, _ = run("COLO-01/v1_postponed", C1_CORRECT)
        users = [m["content"] for m in agent.messages if m["role"] == "user"]
        # 每一則 user 訊息都必須是觀察或 NUDGE，不能有第三種
        for u in users:
            self.assertTrue(u == NUDGE or u.startswith("Today is"), u)

    def test_the_system_prompt_is_exactly_the_file_plus_the_tool_list(self):
        from rsandbox.loaders import prompts
        s, agent, _ = run("COLO-01/v1_postponed", C1_CORRECT)
        want = prompts.system_prompt("referral_agent", DATA["tools"])
        self.assertEqual(agent.messages[0]["content"], want)

    def test_the_tool_list_carries_no_side_effects_or_preconditions(self):
        """side_effects 會直接告訴 Agent 哪個工具會動到哪張表——
        那等於把依賴關係畫給它看。"""
        from rsandbox.loaders import prompts
        text = prompts.system_prompt("referral_agent", DATA["tools"])
        for word in ("side_effect", "precondition", "writes:", "updates:",
                     "snapshot", "handler"):
            self.assertNotIn(word, text)

    def test_the_agent_does_not_expose_finished(self):
        """ReAct agent 不知道任務完成了沒有——那正是要測的東西之一。
        所以主迴圈只能靠「這回合有沒有寫入」判斷，不能問它。"""
        agent = ReActAgent(ReplayBackend([]))
        self.assertFalse(hasattr(agent, "finished"))


#  執行紀錄

class TestRunIO(unittest.TestCase):

    def test_the_terminal_state_alone_is_enough_to_rebuild_the_episode(self):
        """存下來的檔案必須足以在不重跑模型的情況下重建整件事。
        跑一次大模型要幾十分鐘，所以之後不管用什麼方式驗證，
        它吃的都會是這幾個檔案，而不是重新跑一次。
        要能重建的東西有三樣：最終狀態、每一次改動、每一次呼叫。
        """
        import json
        import tempfile
        from pathlib import Path

        from rsandbox.runtime import runio
        s, agent, _ = run("COLO-01/v1_postponed", C1_CORRECT)
        with tempfile.TemporaryDirectory() as tmp:
            rid = runio.run_id("COLO-01/v1_postponed", "react", "llama3.1:8b")
            out = runio.write(rid, {"task": "COLO-01/v1_postponed"}, s, agent.transcript,
                              root=Path(tmp))
            back = runio.read_state(rid, root=Path(tmp))
            with open(out / "episode.json", encoding="utf-8") as f:
                ep = json.load(f)
        # 最終狀態一字不差
        self.assertEqual(back["state"]["hold_orders"], s.state["hold_orders"])
        # 過程也在
        self.assertEqual(len(ep["change_log"]), len(s.state["change_log"]))
        self.assertEqual(len(ep["tool_calls"]), len(s.state["tool_call_log"]))
        # 不含答案鑰匙：公開釋出的時候規則表不隨資料一起走
        self.assertNotIn("knowledge", ep)
        self.assertNotIn("metrics", ep)

    def test_the_run_id_has_no_timestamp(self):
        from rsandbox.runtime import runio
        # 斜線會被換成底線，不然 runs/ 底下會變成兩層目錄
        self.assertEqual(runio.run_id("COLO-01/v1_postponed", "react", "llama3.1:8b"),
                         "COLO-01_v1_postponed_react_llama3.1-8b")
        self.assertEqual(runio.run_id("COLO-01/v1_postponed", "react", "llama3.1:8b", 3),
                         "COLO-01_v1_postponed_react_llama3.1-8b_r3")


if __name__ == "__main__":
    unittest.main()


class TestReadingCountsAsActing(unittest.TestCase):
    """真模型第一次跑就撞到的迴圈 bug（2026-09-30）。
    llama3.1:8b 的第一個回合是
        get_patient_record -> end_turn
    一個讀、一個交還時間。主迴圈原本只把寫入算成「有做事」，
    所以這個回合等於什麼都沒做，episode 在第一回合就結束了——
    baseline 只跑 2 步、連預約都沒建，看起來像「模型太弱」，
    其實是迴圈把它掐死了。

    而且它跟提示詞自相矛盾：
        "When you have nothing further to do for now, call end_turn"
    那個 for now 承諾了還會有下一次。Agent 照著做，卻被當成放棄。

    scripted 軌跡永遠抓不到這個 bug：它們的第一個回合都有寫入，
      而且 ScriptedAgent 有 finished() 走另一條分支。
    """

    READ_THEN_END = [
        ("call", "get_patient_record", {}),
        ("call", "end_turn", {"note": "Reviewed the record."}),
    ]
    THEN_WORK = [
        ("call", "create_referral", {"institution_id": "H1", "dept": "GI",
                                     "service": "COL"}),
        ("call", "book_appointment", {"slot_id": "H1-GI-COL-20260910-AM"}),
        ("call", "end_turn", {}),
    ]

    def test_a_read_only_turn_does_not_end_the_episode(self):
        task = DATA["tasks"]["COLO-01/v0_clean"]
        agent = ReActAgent(ReplayBackend(self.READ_THEN_END + self.THEN_WORK))
        s = loop.run_task(DATA, task, agent)
        self.assertGreaterEqual(s.state["turns"], 2,
                                "讀了病歷就交還時間，episode 卻結束了")
        self.assertIn("APT-001", s.state["appointments"])

    def test_a_completely_empty_turn_does_end_it(self):
        """反面：一整個回合連一次成功呼叫都沒有，才代表它真的沒事做。
        不然 C1 會從 09-03 空轉到 09-30，稀釋 C2 的負控制。"""
        task = DATA["tasks"]["COLO-01/v0_clean"]
        agent = ReActAgent(ReplayBackend([("call", "end_turn", {})]))
        s = loop.run_task(DATA, task, agent)
        self.assertEqual(s.state["turns"], 1)

    def test_end_turn_alone_does_not_count_as_acting(self):
        from rsandbox.runtime.loop import _acted_this_turn
        task = DATA["tasks"]["COLO-01/v1_postponed"]
        agent = ReActAgent(ReplayBackend([("call", "end_turn", {})]))
        s = loop.run_task(DATA, task, agent, BASE)
        own = [r for r in s.state["tool_call_log"] if r["at"] >= "2026-09-03"]
        self.assertEqual([r["tool"] for r in own], ["end_turn"])
        self.assertEqual(s.state["turns"], 1)

    def test_a_failed_call_does_not_count_as_acting(self):
        """只有成功的呼叫算在做事。一個回合全是失敗呼叫，
        代表它在撞牆而不是在推進——但那也該結束，不然會無限撞。"""
        task = DATA["tasks"]["COLO-01/v0_clean"]
        agent = ReActAgent(ReplayBackend([
            ("call", "book_appointment", {"slot_id": "NOPE"}),
            ("call", "end_turn", {})]))
        s = loop.run_task(DATA, task, agent)
        self.assertEqual(s.state["turns"], 1)


class TestTranscriptIsSelfContained(unittest.TestCase):
    """讀 transcript 是診斷模型行為的主要手段，所以它要能獨立閱讀，
    不用跟 state.json 對照。"""

    def test_it_records_observations_calls_and_results(self):
        s, agent, _ = run("COLO-01/v1_postponed", C1_CORRECT)
        kinds = [set(row) - {"t"} for row in agent.transcript]
        self.assertIn({"observation"}, kinds)
        self.assertIn({"message"}, kinds)
        self.assertTrue(any("result" in row for row in agent.transcript))

    def test_a_failed_call_records_its_reason(self):
        s, agent, _ = run("COLO-01/v1_postponed", [("call", "book_appointment",
                                  {"slot_id": "H1-GI-COL-20260910-AM"}),
                                 ("call", "end_turn", {})])
        row = next(r for r in agent.transcript if r.get("tool") == "book_appointment")
        self.assertFalse(row["ok"])
        self.assertEqual(row["reason"], "capacity_full")


class TestUnparsedToolCalls(unittest.TestCase):
    """工具呼叫被寫成 content 裡的文字 —— 偵測、計數，但不執行。
    2026-09-30 llama3.1:8b 第一批實驗發現的：模型先講一段散文再吐
    `{"name": ..., "parameters": ...}`，Ollama 的 chat template parser 抓不到，
    那個呼叫就留在 content 裡，從來沒有被執行。

    為什麼要測這個：不計數的話，「模型不會用工具」跟「模型的呼叫被吃掉了」
    在失敗統計裡長得一模一樣，而原生 function calling 的全部意義就是要
    消滅這種雜訊。
    而為什麼不能執行：去 content 撈 JSON 出來跑，就是退回文字協定。
    """

    SCRIPT = [("say", 'Let me check the status. '
                      '{"name": "get_case_status", "parameters": {}}'),
              ("call", "end_turn", {})]

    def test_it_is_counted(self):
        s, agent, _ = run("COLO-01/v1_postponed", self.SCRIPT, nudges=1)
        self.assertEqual([u["name"] for u in agent.unparsed], ["get_case_status"])
        self.assertFalse(agent.unparsed[0]["had_native_call"])

    def test_it_is_not_executed(self):
        """最關鍵的一條：那個工具沒有進 tool_call_log。"""
        s, agent, _ = run("COLO-01/v1_postponed", self.SCRIPT, nudges=1)
        own = [r["tool"] for r in s.state["tool_call_log"] if r["at"] >= "2026-09-03"]
        self.assertNotIn("get_case_status", own)

    def test_it_goes_into_the_transcript(self):
        s, agent, _ = run("COLO-01/v1_postponed", self.SCRIPT, nudges=1)
        rows = [r for r in agent.transcript if "unparsed_tool_call" in r]
        self.assertEqual([r["unparsed_tool_call"] for r in rows], ["get_case_status"])

    def test_it_is_counted_even_alongside_a_native_call(self):
        """同一則訊息既有原生呼叫、content 裡又有一個 JSON：
        代表模型想做兩件事，其中一件被 parser 吃掉了。那一件也要記。"""
        s, agent, _ = run("COLO-01/v1_postponed", [
            ("raw", {"role": "assistant",
                     "content": 'Also {"name": "lookup_hold", "parameters": {}}',
                     "tool_calls": [{"id": "c1", "name": "get_case_status",
                                     "arguments": {}}]}),
            ("call", "end_turn", {})])
        self.assertEqual([u["name"] for u in agent.unparsed], ["lookup_hold"])
        self.assertTrue(agent.unparsed[0]["had_native_call"])
        own = [r["tool"] for r in s.state["tool_call_log"] if r["at"] >= "2026-09-03"]
        self.assertIn("get_case_status", own)      # 原生的那個照跑
        self.assertNotIn("lookup_hold", own)       # 文字的那個不跑

    def test_ordinary_prose_is_not_flagged(self):
        """反面：偵測器不能把正常的散文或提到 JSON 的句子算進去。
        一個永遠會叫的偵測器等於沒有偵測器。"""
        s, agent, _ = run("COLO-01/v1_postponed", [
            ("say", "I updated the phone number. The name field is a string."),
            ("call", "end_turn", {})], nudges=1)
        self.assertEqual(agent.unparsed, [])


class TestDroppedExtraCalls(unittest.TestCase):
    """one 模式（預設）：同一則訊息第二個以後的呼叫不執行，但要計數。"""

    def test_one_is_the_default(self):
        """不給 tool_calls 的時候是 one。"""
        self.assertEqual(ReActAgent(ReplayBackend([])).tool_calls, "one")

    def test_an_unknown_mode_is_refused(self):
        with self.assertRaises(ValueError):
            ReActAgent(ReplayBackend([]), tool_calls="parallel")

    def test_extras_are_counted_and_not_executed(self):
        task = DATA["tasks"]["COLO-01/v1_postponed"]
        agent = ReActAgent(ReplayBackend([
            ("raw", {"role": "assistant", "content": "",
                     "tool_calls": [
                         {"id": "a", "name": "update_patient_contact",
                          "arguments": {"phone": "0912-345-678"}},
                         {"id": "b", "name": "end_turn", "arguments": {}}]}),
            ("call", "end_turn", {})]))
        s = loop.run_task(DATA, task, agent, BASE)
        self.assertEqual([d["name"] for d in agent.dropped], ["end_turn"])
        self.assertEqual(agent.dropped[0]["after"], "update_patient_contact")
        first = [r for r in s.state["tool_call_log"] if r["at"] == "2026-09-03"]
        # 第一個跑了，第二個沒有——end_turn 只出現在下一步（腳本的第二項）
        self.assertEqual(first[0]["tool"], "update_patient_contact")

    def test_the_transcript_marks_which_call_ran(self):
        """沒有這一欄，show-run 的「呼叫│」與「結果│」會對不齊。"""
        task = DATA["tasks"]["COLO-01/v1_postponed"]
        agent = ReActAgent(ReplayBackend([
            ("raw", {"role": "assistant", "content": "",
                     "tool_calls": [
                         {"id": "a", "name": "get_case_status", "arguments": {}},
                         {"id": "b", "name": "end_turn", "arguments": {}}]}),
            ("call", "end_turn", {})]))
        loop.run_task(DATA, task, agent, BASE)
        row = next(r for r in agent.transcript
                   if r.get("message", {}).get("tool_calls"))
        self.assertEqual([c["executed"] for c in row["message"]["tool_calls"]],
                         [True, False])


#  batch 模式

BATCH = [("raw", {"role": "assistant", "content": "",
                  "tool_calls": [
                      {"id": "a", "name": "get_case_status", "arguments": {}},
                      {"id": "b", "name": "get_patient_record", "arguments": {}},
                      {"id": "c", "name": "list_institutions", "arguments": {}}]}),
         ("call", "end_turn", {})]


class TestBatchMode(unittest.TestCase):
    """batch 模式：整批依序執行，每一個的結果都送回去。"""

    def test_every_call_in_the_batch_runs_in_the_listed_order(self):
        s, agent, _ = run("COLO-01/v1_postponed", BATCH, tool_calls="batch")
        own = [r["tool"] for r in s.state["tool_call_log"]
               if r["at"] >= "2026-09-03"]
        self.assertEqual(own[:3], ["get_case_status", "get_patient_record",
                                   "list_institutions"])
        self.assertEqual(agent.dropped, [])

    def test_every_call_in_the_batch_gets_a_real_result(self):
        """一個都不能收到 EXTRA_CALL：那一則的意思是沒有執行。"""
        _, agent, _ = run("COLO-01/v1_postponed", BATCH, tool_calls="batch")
        notes = [json.loads(m["content"]) for m in agent.messages
                 if m["role"] == "tool"]
        self.assertNotIn(EXTRA_CALL, notes)

    def test_the_transcript_marks_the_whole_batch_as_executed(self):
        _, agent, _ = run("COLO-01/v1_postponed", BATCH, tool_calls="batch")
        row = next(r for r in agent.transcript
                   if r.get("message", {}).get("tool_calls"))
        self.assertEqual([c["executed"] for c in row["message"]["tool_calls"]],
                         [True, True, True])

    def test_calls_after_end_turn_do_not_run(self):
        """排在 end_turn 後面的呼叫不執行，但要進 dropped。"""
        s, agent, _ = run("COLO-01/v1_postponed", [
            ("raw", {"role": "assistant", "content": "",
                     "tool_calls": [
                         {"id": "a", "name": "get_case_status", "arguments": {}},
                         {"id": "b", "name": "end_turn", "arguments": {}},
                         {"id": "c", "name": "update_patient_contact",
                          "arguments": {"phone": "0912-345-678"}}]})],
            tool_calls="batch")
        own = [r["tool"] for r in s.state["tool_call_log"]
               if r["at"] == "2026-09-03"]
        self.assertEqual(own, ["get_case_status", "end_turn"])
        self.assertEqual([d["name"] for d in agent.dropped],
                         ["update_patient_contact"])
        self.assertEqual(agent.dropped[0]["after"], "end_turn")

    def test_the_prompt_tells_the_model_which_mode_it_is_in(self):
        """提示詞裡的呼叫規則要跟著模式換，其餘逐字相同。"""
        from rsandbox.loaders.prompts import CALL_RULE, prompt_text
        one = prompt_text("referral_agent")
        batch = prompt_text("referral_agent", tool_calls="batch")
        self.assertIn(CALL_RULE["one"], one)
        self.assertNotIn(CALL_RULE["one"], batch)
        self.assertIn(CALL_RULE["batch"], batch)
        # 換掉的只有那一行，其餘逐字相同。
        self.assertEqual(one.replace(CALL_RULE["one"], CALL_RULE["batch"]), batch)

    def test_the_rule_line_matches_the_prompt_file(self):
        """每一個模式都組得出提示詞；CALL_RULE 與提示詞檔對不上就丟錯。"""
        from rsandbox.loaders.prompts import prompt_text
        for mode in TOOL_CALL_MODES:
            with self.subTest(mode=mode):
                prompt_text("referral_agent", tool_calls=mode)


class TestRecordAssessmentIsNotWorldProgress(unittest.TestCase):
    """record_assessment 寫的是 Agent 自己的記事本，不是世界。
    算成「推進世界」的話，一個每回合記一筆「還在等結果」的模型永遠不會被
    IDLE_TURN_LIMIT 攔下來。qwen2.5:7b 的 C4 就這樣跑了 27 個回合、79 步。
    """

    def test_polling_with_record_assessment_gets_stopped(self):
        task = DATA["tasks"]["COLO-01/v4_slot_cancelled"]
        poll = [("call", "get_case_status", {}),
                ("call", "record_assessment", {"key": "waiting", "value": "yes"}),
                ("call", "end_turn", {})] * 12
        agent = ReActAgent(ReplayBackend(poll))
        s = loop.run_task(DATA, task, agent, BASE)
        self.assertLessEqual(s.state["turns"], loop.IDLE_TURN_LIMIT + 1,
                             "只讀 + 記事本的回合被當成推進世界了")

    def test_record_assessment_is_not_in_write_tools(self):
        self.assertNotIn("record_assessment", loop.WRITE_TOOLS)


class TestTheTranscriptIsAFrozenRecord(unittest.TestCase):
    """transcript 是「模型當下看到什麼」的紀錄，必須凍結。
    2026-10-13 抓到的 bug：工具回傳的結構裡，巢狀值（預約的 history 清單）
    跟 state 裡的是同一個物件。transcript 直接存參考的話，之後
    reschedule 往 history append 一筆，早就寫好的那一則紀錄會跟著變 ——
    讀 show-raw 的人會看到「模型當時看到了它其實還沒看到的東西」。

    模型收到的那份沒事（_tool_reply 立刻 json.dumps 成字串），
      評分器讀的 tool_call_log 也沒事（Session._log 有深拷貝）。
      壞的只有 transcript，也就是人在看的那一份 —— 而那正是
      診斷模型行為的主要依據。
    """

    def test_a_tool_result_does_not_change_after_the_fact(self):
        script = [("call", "get_case_status", {}),
                  ("call", "reschedule_appointment",
                   {"appointment_id": "APT-001",
                    "new_slot_id": "H1-GI-COL-20260917-AM"}),
                  ("call", "end_turn", {})]
        _s, agent, _m = run("COLO-01/v1_postponed", script)
        row = next(r for r in agent.transcript if r.get("tool") == "get_case_status")
        ap = row["result"]["appointments"][0]
        # 呼叫 get_case_status 的當下還沒改期，所以 history 必須是空的
        self.assertEqual(ap["history"], [],
                         "transcript 裡的紀錄被之後的改期事後竄改了")
        self.assertEqual(ap["date"], "2026-09-10")

    def test_the_args_are_frozen_too(self):
        """參數也要拷貝：handler 可能就地改寫傳進去的 dict。"""
        script = [("call", "search_guidelines", {"query": "blood test"}),
                  ("call", "end_turn", {})]
        _s, agent, _m = run("COLO-01/v1_postponed", script)
        row = next(r for r in agent.transcript
                   if r.get("tool") == "search_guidelines")
        self.assertEqual(row["args"], {"query": "blood test"})


#  bad_arguments 要說是哪一個參數

class TestBadArgumentsNamesTheParameter(unittest.TestCase):
    """bad_arguments 只回理由的時候，模型沒有任何修正依據，於是重送同一個
    呼叫。fields 說哪幾個參數有問題。
    界線：說欄位名與問題類型，不說傳進來的值、不說合法值清單。
    """

    def envelope(self, tool, args):
        from rsandbox.runtime import initial
        s = initial.build(DATA, DATA["tasks"]["COLO-01/v0_clean"])
        return s.call(tool, args)

    def payload(self, tool, args):
        from rsandbox.agents.react import _error_payload
        env = self.envelope(tool, args)
        self.assertFalse(env["ok"], env)
        return _error_payload(env)

    def test_an_invented_parameter_is_named(self):
        p = self.payload("query_slots",
                         {"institution_id": "H1", "dept": "GI", "service": "COL",
                          "date_from": "2026-09-01", "date_to": "2026-09-30",
                          "weekdays": ["Mon"]})
        self.assertEqual(p["error"], "bad_arguments")
        self.assertEqual(p["fields"], ["unknown parameter 'weekdays'"])

    def test_every_missing_required_parameter_is_named(self):
        p = self.payload("query_slots", {"institution_id": "H1", "dept": "GI"})
        self.assertEqual(p["error"], "bad_arguments")
        self.assertEqual(len(p["fields"]), 3)
        for name in ("service", "date_from", "date_to"):
            with self.subTest(param=name):
                self.assertTrue(any(f"'{name}'" in f for f in p["fields"]), p["fields"])

    def test_the_message_says_only_what_is_wrong(self):
        """只說欄位名與問題類型，不接參數的描述、不給該填什麼。"""
        from rsandbox.loaders import load_all
        desc = (load_all()["tools"]["query_slots"]["params"]["service"]
                ["description"])
        p = self.payload("query_slots", {"institution_id": "H1", "dept": "GI",
                                         "date_from": "2026-09-01",
                                         "date_to": "2026-09-30"})
        self.assertEqual(p["fields"], ["missing required parameter 'service'"])
        self.assertNotIn(desc, json.dumps(p, ensure_ascii=False))

    def test_a_wrong_type_is_named(self):
        p = self.payload("query_slots",
                         {"institution_id": "H1", "dept": "GI", "service": "COL",
                          "date_from": 20260901, "date_to": "2026-09-30"})
        self.assertIn("date_from", " ".join(p["fields"]))

    def test_fields_never_leaks_the_allowed_values(self):
        """enum 超出範圍的時候不可以列出合法值，那等於把答案印出來。"""
        from rsandbox.loaders import load_all
        d = load_all()
        enums = {v for t in d["tools"].values()
                 for s in (t.get("params") or {}).values()
                 for v in (s.get("enum") or [])}
        self.assertTrue(enums, "沒有任何參數有 enum，這條測試會變成空的")
        p = self.payload("lookup_hold", {"drug": "warfarin",
                                         "indication_class": "not_a_real_class",
                                         "procedure_category": "endoscopic",
                                         "procedure_risk": "high"})
        text = json.dumps(p, ensure_ascii=False)
        for v in sorted(enums):
            self.assertNotIn(f'"{v}"', text, f"fields 洩漏了合法值 {v!r}")

    def test_fields_never_echoes_what_was_sent(self):
        """傳錯的值不回送：回送等於確認「你猜的這個不在清單上」，
        那是一次二十分之一的篩除，用重送就能窮舉。
        """
        p = self.payload("lookup_hold", {"drug": "warfarin",
                                         "indication_class": "afib_guess_12345",
                                         "procedure_category": "endoscopic",
                                         "procedure_risk": "high"})
        self.assertNotIn("afib_guess_12345", json.dumps(p, ensure_ascii=False))

    def test_a_successful_envelope_has_no_fields_key(self):
        env = self.envelope("query_slots",
                            {"institution_id": "H1", "dept": "GI", "service": "COL",
                             "date_from": "2026-09-01", "date_to": "2026-09-30"})
        self.assertTrue(env["ok"])
        self.assertNotIn("fields", env)

    def test_other_reasons_carry_no_fields(self):
        """fields 是給呼叫方式錯誤用的。世界的合法拒絕不帶它。"""
        env = self.envelope("query_slots",
                            {"institution_id": "H1", "dept": "GI",
                             "service": "NOPE",
                             "date_from": "2026-09-01", "date_to": "2026-09-30"})
        self.assertEqual(env["reason"], "unknown_service")
        self.assertNotIn("fields", env)
