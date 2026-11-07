"""匯出 Agent 看得到的所有字串，並檢查它們。
在沙盒之外執行，沙盒不 import 它。

    python checks/agent_visible.py            匯出成 JSON 並檢查
    python checks/agent_visible.py --dump     只匯出，不檢查
    python checks/agent_visible.py --check out.json   檢查既有的匯出檔

匯出格式是一個清單，每筆 {"where": 來源, "text": 字串}。
「來源」是 檔案路徑:欄位路徑，用來定位要改哪一個檔。

收集範圍：
    data/prompts/*.txt                  Agent 的系統提示詞
    各元素 guidelines/*.yaml            search_guidelines 回傳的欄位
    各組 record.yaml 的每個字串值        get_patient_record 原樣回傳
    每支工具的描述、標籤、參數描述
    每張角色卡上病人會說出來的話
    每個任務的事件內容原文
    runtime/messages.py 的模板表         環境發出的通知

回傳碼 0 = 通過，1 = 有問題。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rsandbox.paths import DATA  # noqa: E402


def _as_text(value) -> str:
    """把嵌套的結構攤平成一行。"""
    if isinstance(value, dict):
        return " ".join(f"{k} {_as_text(v)}" for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return " ".join(_as_text(v) for v in value)
    return str(value)


def agent_visible_strings(root: Path = DATA) -> list[dict]:
    """回傳 [{"where": 來源, "text": 字串}]，照來源排序。"""
    from rsandbox.loaders import cases, graph, guidelines, patients, records
    from rsandbox.loaders.guidelines import for_agent
    from rsandbox.runtime.messages import TEMPLATES

    out: list[dict] = []

    def add(where: str, text) -> None:
        out.append({"where": where, "text": str(text)})

    for p in sorted((root / "prompts").glob("*.txt")):
        add(f"data/prompts/{p.name}", p.read_text(encoding="utf-8"))

    # 呼叫規則那一行不住在提示詞檔裡，是按模式換進去的，所以單獨掃。
    from rsandbox.loaders.prompts import CALL_RULE
    for mode, line in sorted(CALL_RULE.items()):
        add(f"prompts.CALL_RULE[{mode!r}]", line)

    def walk(node, where):
        """往下走遍整個巢狀結構，把每一個字串都撈出來。"""
        if isinstance(node, str):
            add(where, node)
        elif isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{where}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{where}[{i}]")

    for pid, rec in sorted(records.fresh_copy(root).items()):
        walk(rec, f"data/cases/*/record.yaml:{pid}")

    # for_agent 過濾後才是會回傳的欄位；assumption / source 不回傳。
    for gid, g in sorted(guidelines.load_guidelines(root).items()):
        for k, v in sorted(for_agent(g).items()):
            add(f"data/elements/{g['element']}/guidelines/{gid}.yaml:{k}",
                _as_text(v))

    for tid, t in sorted(graph.load_tools(root).items()):
        where = f"data/tools/{t['level']}/{t['_file']}:{tid}"
        add(where, t.get("description", ""))
        for x in t.get("tags") or []:
            add(f"{where}.tags", x)
        for pname, spec in (t.get("params") or {}).items():
            if spec.get("description"):
                add(f"{where}.{pname}", spec["description"])

    # 病人的個人視角資料：這些文字會變成病人說的話，
    #   而病人說的話會進到轉診 Agent 眼前。
    for pid, doc in sorted(patients.load_patient_docs(root).items()):
        where = f"data/cases/*/patient_doc.yaml:{pid}"
        add(f"{where}.persona", doc.get("persona", ""))
        for name, text in sorted((doc.get("knows") or {}).items()):
            add(f"{where}.knows.{name}", text)
        for i, e in enumerate(doc.get("log") or []):
            add(f"{where}.log[{i}]", e["text"])

    # 病人 Agent 的行為規則。它自己讀，不給轉診 Agent 看，
    #   但它會塑造病人說出來的每一句話，所以一起掃。
    from rsandbox.loaders import prompts
    add("data/prompts/patient_agent.txt", prompts.prompt_text("patient_agent", root))

    tasks, _scripts = cases.load_cases(root)
    for tid, task in sorted(tasks.items()):
        for i, ev in enumerate(task.get("event_script") or []):
            add(f"data/cases/{tid}.yaml:event[{i}]", ev.get("content", ""))

    for k, v in sorted(TEMPLATES.items()):
        add(f"runtime/messages.py:{k}", v)

    return out


#  檢查

def check_ascii(rows: list[dict]) -> list[str]:
    """每一個字串都必須是 ASCII。
    模型讀到中文會用中文去查全英文的指引庫，整批查詢落空，
    而那種失敗在逐字稿裡看起來像推理失敗。
    """
    bad = []
    for r in rows:
        for c in r["text"]:
            if ord(c) > 127:
                bad.append(f"{r['where']}：{c!r} 出現在 {r['text'][:60]!r}")
                break
    return bad


def check_coverage(rows: list[dict]) -> list[str]:
    """確認會原樣回傳的兩大來源真的在匯出範圍裡。
    漏收的話上面那條檢查會變成永遠通過。
    """
    where = [r["where"] for r in rows]
    bad = []
    if not any("record.yaml:P001" in w for w in where):
        bad.append("病歷不在匯出範圍裡（get_patient_record 原樣回傳它）")
    if not any("guidelines/" in w for w in where):
        bad.append("指引不在匯出範圍裡（search_guidelines 原樣回傳它）")
    return bad


CHECKS = (("ASCII", check_ascii), ("覆蓋範圍", check_coverage))


def run_checks(rows: list[dict]) -> int:
    errs = []
    for name, fn in CHECKS:
        found = fn(rows)
        print(f"  {name:8} {'✓' if not found else f'✗ {len(found)} 處'}")
        errs += found
    print()
    for e in errs:
        print(f"  ✗ {e}")
    print(f"{'✗ ' + str(len(errs)) + ' 個問題' if errs else '✓ 通過'}"
          f"（掃了 {len(rows)} 個字串）")
    return 1 if errs else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="store_true", help="只匯出，不檢查")
    ap.add_argument("--check", metavar="JSON", help="檢查既有的匯出檔")
    ap.add_argument("-o", default=str(Path(__file__).parent / "agent_visible.json"),
                    help="匯出檔名")
    a = ap.parse_args()

    if a.check:
        rows = json.loads(Path(a.check).read_text(encoding="utf-8"))
        print(f"讀 {a.check}")
        return run_checks(rows)

    rows = agent_visible_strings()
    Path(a.o).write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"匯出 {len(rows)} 個字串 → {a.o}")
    if a.dump:
        return 0
    print()
    return run_checks(rows)


if __name__ == "__main__":
    raise SystemExit(main())
