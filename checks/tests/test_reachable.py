"""可達性：正解傳出去的每一個值，能不能從起點查出來。
流程：
  - 從起點把讀取類工具窮舉一遍，拿到的代號再往下一層呼叫
  - 收集所有回傳過的字串，得到可見集合
  - 拿正解每一步傳的參數值去比對，不在集合裡的列出來
"""
from __future__ import annotations

import unittest

from rsandbox.agents.scripted import ScriptedAgent
from rsandbox.loaders import load_all
from rsandbox.runtime import initial, loop

DATA = load_all()

# FREE（Agent 自己寫的參數，不是查來的），不列入比對。
FREE = frozenset({"note", "question", "query", "value", "notes", "summary",
                  "points", "key", "content", "evidence", "rule_id", "top_k"})

# PROBE_QUERIES（窮舉指引檢索用的關鍵詞）。
PROBE_QUERIES = ("colonoscopy", "warfarin", "blood test", "bleeding risk",
                 "screening", "anticoagulant", "hold")


def flatten(x, out: set) -> set:
    """flatten（把巢狀結構裡的每一個純量撈成字串）。"""
    if isinstance(x, str):
        out.add(x)
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        out.add(str(x))
    elif isinstance(x, dict):
        for v in x.values():
            flatten(v, out)
    elif isinstance(x, (list, tuple)):
        for v in x:
            flatten(v, out)
    return out


def base_snapshot(task_id: str):
    """base_snapshot（這個變體的起始狀態），from: empty 的回 None。"""
    task = DATA["tasks"][task_id]
    if (task.get("initial_state") or {}).get("from") == "empty":
        return None
    return initial.clean_end(DATA, task_id.split("/")[0])


def visible_from_start(task_id: str) -> set:
    """visible_from_start（從起點窮舉讀取類工具，回傳看得到的所有字串）。
    涵蓋的層：
      - 事件宣告的原文、系統提示詞
      - 病歷、案件狀態、指引檢索
      - 機構 -> 科別 -> 項目 -> 診次，逐層展開
    """
    task = DATA["tasks"][task_id]
    s = initial.build(DATA, task, base_snapshot(task_id))
    pool: set = set()
    for ev in task.get("event_script") or []:
        flatten(ev.get("content"), pool)
    from rsandbox.loaders import prompts
    flatten(prompts.system_prompt("referral_agent", s.tools), pool)

    def read(tool: str, **args):
        env = s.call(tool, args)
        if env["ok"]:
            flatten(env["data"], pool)
            return env["data"]
        return None

    read("get_patient_record")
    read("get_case_status")
    for q in PROBE_QUERIES:
        read("search_guidelines", query=q)

    read("list_institutions")

    # 往下展開的機構：代號必須已經出現在可見集合裡，不寫死任何代號。
    world = DATA["worlds"][task_id.split("/")[0]]
    for iid in sorted(world["institutions"]):
        if not any(iid in p for p in pool):
            continue
        info = read("get_institution_info", institution_id=iid) or {}
        for drow in info.get("departments") or []:
            dept = drow.get("dept") or drow.get("code")
            if not dept:
                continue
            dinfo = read("get_department_info",
                         institution_id=iid, dept=dept) or {}
            for srow in dinfo.get("services") or []:
                svc = srow.get("service") or srow.get("code")
                if not svc:
                    continue
                read("get_service_info", institution_id=iid, dept=dept,
                     service=svc)
                read("query_slots", institution_id=iid, dept=dept, service=svc,
                     date_from=task["clock"]["start_date"],
                     date_to=task["clock"]["end_date"])
    return pool


def unreachable(task_id: str) -> list:
    """unreachable（正解傳了、但從起點查不到的值），回 [(第幾步, 工具, 參數, 值)]。
    Agent 自己開出來的單號在它開出來之後算看得見。
    """
    steps = DATA["scripts"][task_id]["reference"]
    agent = ScriptedAgent(steps)
    loop.run_task(DATA, DATA["tasks"][task_id], agent, base_snapshot(task_id))
    pool = visible_from_start(task_id)
    out = []
    for i, r in enumerate(agent.results, 1):
        for name, val in (r.get("args") or {}).items():
            if name in FREE:
                continue
            for v in flatten(val, set()):
                if v in pool or any(v in p for p in pool if len(p) > len(v)):
                    continue
                out.append((i, r["tool"], name, v))
        flatten(r.get("data"), pool)
    return out


class TestEveryValueTheAnswerNeedsIsReachable(unittest.TestCase):
    """每一個變體的正解，傳出去的值都要能從起點查到。
    推論與算術算出來的值（停藥末次日、抽血日）本身也會落在這個集合裡，
      因為它們的素材（手術日、指引的天數）都查得到，而日期字串本身
      出現在診次清單裡。
    """

    def test_every_variant(self):
        for tid in sorted(DATA["scripts"]):
            with self.subTest(task=tid):
                miss = unreachable(tid)
                self.assertEqual(
                    miss, [],
                    "正解用到查不出來的值：\n"
                    + "\n".join(f"  第 {i} 步 {tool}.{name} = {v!r}"
                                for i, tool, name, v in miss))

    def test_the_closure_is_not_empty(self):
        """可見集合要含這幾個代號，確認窮舉真的有逐層展開。"""
        pool = visible_from_start("COLO-01/v0_clean")
        for v in ("H1", "GI", "COL", "C0", "P001"):
            with self.subTest(value=v):
                self.assertIn(v, pool)


if __name__ == "__main__":
    unittest.main()
