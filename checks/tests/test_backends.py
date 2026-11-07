"""模型後端的正規化（agents/backends.py）。
這一層跟資料與世界完全無關，所以第一階段改寫沒有動到它。
保留的原因：Ollama 與 OpenAI 相容介面的 tool call 格式不一樣
（前者 arguments 是物件，後者是 JSON 字串；後者的 tool 結果還要帶
tool_call_id），這些差異必須關在 backends.py 裡，
不能漏到上層——上層只認一種「正規化訊息」。

被測 Agent 本身（ReAct loop）在第 ⑥ 步才寫，它的測試屆時另開一個檔。
"""
import json
import unittest

from rsandbox.agents.backends import (OllamaBackend, OpenAICompatBackend,
                                      from_function_name, to_function_name)


class TestNameMapping(unittest.TestCase):
    def test_round_trip(self):
        # 第一階段的 17 個工具名裡沒有點號，但第二階段會有
        # （H1_GI_query_endoscopy_slots 之類），所以這一層要留著。
        for tid in ("end_turn", "get_patient_record", "H1_GI.book_slot"):
            self.assertEqual(from_function_name(to_function_name(tid)), tid)

    def test_function_names_are_api_safe(self):
        import re
        from rsandbox.loaders.graph import load_tools
        for tid in load_tools():
            self.assertRegex(to_function_name(tid), r"^[A-Za-z0-9_-]{1,64}$")


class TestSerialisation(unittest.TestCase):
    """兩種 provider 的訊息格式不同，序列化要各自正確。"""

    HISTORY = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "obs"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "c1", "name": "CLINIC_FM.get_patient_record",
                         "arguments": {"patient_id": "P001"}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "CLINIC_FM.get_patient_record",
         "content": '{"record": {}}'},
    ]

    def test_ollama_keeps_arguments_as_an_object(self):
        out = OllamaBackend("m").serialize(self.HISTORY)
        call = out[2]["tool_calls"][0]["function"]
        self.assertEqual(call["name"], "CLINIC_FM__get_patient_record")
        self.assertIsInstance(call["arguments"], dict)
        self.assertNotIn("tool_call_id", out[3])

    def test_openai_stringifies_arguments_and_keeps_the_id(self):
        out = OpenAICompatBackend("m").serialize(self.HISTORY)
        call = out[2]["tool_calls"][0]
        self.assertEqual(call["type"], "function")
        self.assertIsInstance(call["function"]["arguments"], str)
        self.assertEqual(json.loads(call["function"]["arguments"]), {"patient_id": "P001"})
        self.assertEqual(out[3]["tool_call_id"], "c1")


if __name__ == "__main__":
    unittest.main()


class TestToolSchemas(unittest.TestCase):
    """上一版的專案在這裡浪費了一輪真模型實驗：list / object 參數沒有展開
    內部結構，模型無從得知裡面該放什麼，於是量到的是「猜不猜得到欄位名」
    而不是「會不會用工具」。這一組測試把那個教訓釘住。"""

    @classmethod
    def setUpClass(cls):
        from rsandbox.agents.schema import all_tool_schemas
        from rsandbox.loaders import load_all
        cls.tools = load_all()["tools"]
        cls.schemas = all_tool_schemas(cls.tools)

    def test_every_tool_produces_a_schema(self):
        """每一支工具都要有 schema，數量與名字都要對得上。"""
        self.assertEqual(len(self.schemas), len(self.tools))
        self.assertEqual({s["function"]["name"] for s in self.schemas},
                         {to_function_name(t) for t in self.tools})

    def test_names_are_api_safe(self):
        import re
        for s in self.schemas:
            self.assertRegex(s["function"]["name"], r"^[A-Za-z0-9_-]{1,64}$")

    def test_every_list_param_declares_its_items(self):
        """沒有 items 的 array 等於對模型說「放個清單」卻不說放什麼。"""
        for s in self.schemas:
            for name, p in s["function"]["parameters"]["properties"].items():
                if p["type"] == "array":
                    self.assertTrue(p.get("items"),
                                    f"{s['function']['name']}.{name} 沒有宣告 items")

    def test_every_object_param_declares_its_properties(self):
        for s in self.schemas:
            for name, p in s["function"]["parameters"]["properties"].items():
                if p["type"] == "object":
                    self.assertTrue(p.get("properties"),
                                    f"{s['function']['name']}.{name} 沒有宣告 properties")

    def test_enums_survive_into_the_schema(self):
        """enum 是從世界資料複製來的（單一事實來源），所以它必須一路傳到
        模型看得到的地方——不然模型不知道 dept_id 只能填那七個。

        2026-10-13：五機構世界之後，enum 參數在 institution_id 上。
          dept 與 service 刻意沒有 enum —— 它們太多、而且因機構而異，
          列出來等於把整個世界的目錄塞進每一次呼叫。Agent 要自己走過去查。
          機制沒變，測的對象換了。
        """
        qs = next(s for s in self.schemas
                  if s["function"]["name"] == "query_slots")
        inst = qs["function"]["parameters"]["properties"]["institution_id"]
        self.assertEqual(inst["enum"], ["C0", "H1", "H2", "H3", "H4"])

    def test_the_guideline_search_has_no_enum_to_lean_on(self):
        """反面：`search_guidelines` 的查詢詞是自由文字，沒有 enum。
        這正是跟原本四支查表工具的關鍵差別 ——
        查表工具的 enum 等於把「有哪幾種用藥原因分類」列給模型看，
        而且工具替它挑好了唯一那一條。
        現在它得自己組查詢詞，再從撈回來的幾份裡比對條件。

        哪天有人給查詢詞加上 enum，這一條會叫。
        """
        sg = next(s for s in self.schemas
                  if s["function"]["name"] == "search_guidelines")
        props = sg["function"]["parameters"]["properties"]
        self.assertNotIn("enum", props["query"])
        self.assertEqual(sg["function"]["parameters"]["required"], ["query"])

    def test_dates_say_they_are_dates(self):
        ho = next(s for s in self.schemas
                  if s["function"]["name"] == "issue_hold_order")
        ld = ho["function"]["parameters"]["properties"]["last_dose_date"]
        self.assertEqual(ld["type"], "string")
        self.assertIn("YYYY-MM-DD", ld["description"])

    def test_required_matches_the_tool_definition(self):
        for s in self.schemas:
            tid = s["function"]["name"]
            want = sorted(n for n, p in (self.tools[tid].get("params") or {}).items()
                          if p.get("required"))
            self.assertEqual(sorted(s["function"]["parameters"]["required"]), want, tid)

    def test_the_schema_leaks_nothing(self):
        """schema 是 Agent 看得到的東西，所以它也受洩漏掃描的判準管。
        handler / precondition / side_effects 一個都不能出現。"""
        import json
        blob = json.dumps(self.schemas, ensure_ascii=False)
        for word in ("handler", "precondition", "side_effects", "runtime.handlers",
                     "recalculat", "depends on"):
            self.assertNotIn(word, blob)


#  傳輸失敗：每一種都要變成 BackendError

class TestEveryTransportFailureBecomesBackendError(unittest.TestCase):
    """每一種傳輸失敗都要變成 BackendError，不能讓例外往上穿。
    讀取逾時單獨測：TimeoutError 與 URLError 都是 OSError 的子類但
      互不相屬，攔 URLError 攔不到它。
    """

    def _raising(self, exc):
        """把 urlopen 換成一個只會丟例外的東西。"""
        from unittest import mock
        from rsandbox.agents import backends
        return mock.patch.object(backends.urllib.request, "urlopen",
                                 side_effect=exc)

    def test_a_read_timeout_becomes_a_backend_error(self):
        from rsandbox.agents.backends import BackendError, OllamaBackend
        with self._raising(TimeoutError("timed out")):
            with self.assertRaises(BackendError) as cm:
                OllamaBackend("m", timeout=7).chat([], [])
        self.assertIn("7", str(cm.exception))

    def test_other_os_errors_become_backend_errors_too(self):
        from rsandbox.agents.backends import BackendError, OllamaBackend
        for exc in (ConnectionResetError("reset"), OSError("broken")):
            with self.subTest(exc=type(exc).__name__):
                with self._raising(exc):
                    with self.assertRaises(BackendError):
                        OllamaBackend("m").chat([], [])

    def test_the_episode_survives_a_timeout(self):
        """後端失敗記進 agent.errors，狀態留著，不往上丟例外。"""
        from unittest import mock

        from rsandbox.agents import backends
        from rsandbox.agents.react import ReActAgent
        from rsandbox.loaders import load_all
        from rsandbox.runtime import initial, loop
        data = load_all()
        task = data["tasks"]["COLO-01/v1_postponed"]
        agent = ReActAgent(backends.OllamaBackend("m"))
        with mock.patch.object(backends.urllib.request, "urlopen",
                               side_effect=TimeoutError("timed out")):
            s = loop.run_task(data, task, agent,
                              initial.clean_end(data, "COLO-01"))
        self.assertTrue(agent.errors, "逾時沒有被記成後端錯誤")
        self.assertIsNotNone(s.state)


#  context 上限

class TestContextLimitIsVisible(unittest.TestCase):
    """每一次呼叫都要記下送出量與服務端處理量。
    服務端切掉超出上限的部分時不會報錯，呼叫紀錄上沒有痕跡。
    """

    def _answering(self, body: dict):
        import io
        from unittest import mock
        from rsandbox.agents import backends
        payload = json.dumps(body).encode()

        class R(io.BytesIO):
            def __enter__(self): return self
            def __exit__(self, *a): return False

        return mock.patch.object(backends.urllib.request, "urlopen",
                                 return_value=R(payload))

    REPLY = {"message": {"content": "hi", "tool_calls": []},
             "prompt_eval_count": 4096}

    def test_num_ctx_reaches_the_request(self):
        from unittest import mock
        from rsandbox.agents import backends
        with self._answering(self.REPLY) as m:
            backends.OllamaBackend("m", num_ctx=8192).chat(
                [{"role": "user", "content": "hi"}], [])
        sent = json.loads(m.call_args[0][0].data)
        self.assertEqual(sent["options"]["num_ctx"], 8192)

    def test_no_num_ctx_sends_no_num_ctx(self):
        """給 0（→ None）的時候 options 裡不能有 num_ctx。"""
        from rsandbox.agents import backends
        with self._answering(self.REPLY) as m:
            backends.OllamaBackend("m").chat(
                [{"role": "user", "content": "hi"}], [])
        sent = json.loads(m.call_args[0][0].data)
        self.assertNotIn("num_ctx", sent["options"])

    def test_both_sizes_are_recorded(self):
        from rsandbox.agents import backends
        b = backends.OllamaBackend("m", num_ctx=8192)
        with self._answering(self.REPLY):
            b.chat([{"role": "user", "content": "hi"}], [])
        row = b.usage_rows()[0]
        self.assertEqual(row["prompt_eval_count"], 4096)
        self.assertEqual(row["num_ctx"], 8192)
        self.assertGreater(row["sent_tokens"], 0)

    def test_the_warning_fires_only_when_the_limit_is_known(self):
        from rsandbox.cli import _ctx_note
        rows = [{"sent_tokens": 9000}, {"sent_tokens": 3000}]
        self.assertIn("1 次", _ctx_note(rows, 8192))
        self.assertEqual(_ctx_note(rows, 8192 * 2), "")
        # 上限未知的時候不下判定：服務端的預設是多少這裡不知道。
        self.assertEqual(_ctx_note(rows, None), "")


#  同一次執行裡的每一個後端都要用同一個 num_ctx

class TestEveryBackendInARunAgreesOnNumCtx(unittest.TestCase):
    """同一個模型用兩個不同的 context 大小，服務端會在兩邊來回的時候
    反覆卸載重載，結果是逾時與殘缺的回應。
    被測 Agent 與病人各建一個後端，兩個都要吃到命令列給的值。
    """

    class Args:
        provider = "ollama"
        model = "m"
        base_url = None
        api_key = None
        seed = 0
        temperature = 0.0
        num_ctx = 16384
        patient_model = "m"
        patient_provider = None
        patient_base_url = None
        patient_api_key = None

    def test_the_patient_backend_gets_the_same_num_ctx(self):
        from rsandbox.agents.backends import build_backend
        from rsandbox.cli import _patient_backend
        a = self.Args()
        agent = build_backend(a.provider, a.model, a.base_url, a.api_key,
                              seed=a.seed, temperature=a.temperature,
                              num_ctx=a.num_ctx)
        patient = _patient_backend(a)
        self.assertEqual(patient.num_ctx, agent.num_ctx)

    def test_zero_means_server_default_on_both_sides(self):
        from rsandbox.cli import _patient_backend
        a = self.Args()
        a.num_ctx = 0
        self.assertIsNone(_patient_backend(a).num_ctx)

    def test_no_patient_model_needs_no_backend(self):
        from rsandbox.cli import _patient_backend
        a = self.Args()
        a.patient_model = "none"
        self.assertIsNone(_patient_backend(a))

    def test_every_subcommand_that_builds_a_backend_offers_the_flag(self):
        """會組後端的子指令都要有 --num-ctx，缺的那一個會用回服務端預設。"""
        import contextlib
        import io

        from rsandbox.cli import main
        for name in ("run", "trace", "check-llm"):
            with self.subTest(cmd=name):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    with self.assertRaises(SystemExit):
                        main([name, "--help"])
                self.assertIn("--num-ctx", out.getvalue())
