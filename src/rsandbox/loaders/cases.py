"""載入任務：一個資料夾一組，一組底下N(5)個變體。
目錄：
    data/cases/COLO-01/            一組 = 一個臨床情境
      case.yaml                    整組共用：病人、起點、時鐘
      record.yaml                  這一組的病歷
      patient_sim.yaml             這一組的病人角色卡
      v0_clean.yaml                無擾動。其他變體接在它跑完的狀態上
      v1_postponed.yaml            病人要求延後
      …
      scripts/                     腳本，跟任務分開
        v0_clean.yaml
        v1_postponed.yaml

世界由 case.yaml 的 elements（必要元素）與 distractors（干擾元素）
兩份清單組出來，兩者的聯集就是這一組載入的元素。

manifest 是設計紀錄：預設標準答案中這一題需要哪些指引、工具、項目。
程式只是把它讀進來帶著走。
"""
from __future__ import annotations

import copy
from pathlib import Path

from ..paths import DATA, load_yaml, yaml_files

# ADVANCE_ON：時鐘可以在什麼時機前進。目前只有「Agent 結束回合」這一種。
ADVANCE_ON = ("end_turn",)

# EVENT_TYPES：擾動事件有哪幾種來源。
EVENT_TYPES = ("patient_call", "system_notice", "system_reminder")

# CASE_KEYS / VARIANT_KEYS：兩個層級各自允許的欄位。
CASE_KEYS = frozenset({"case_id", "title", "patient_id", "start_node", "clock",
                       "manifest", "elements", "distractors"})
VARIANT_KEYS = frozenset({"variant", "title", "initial_state", "clock",
                          "event_script", "manifest", "start_node",
                          "patient_doc_patch"})

# 變體可以覆寫的時鐘欄位。
CLOCK_KEYS = frozenset({"start_date", "end_date", "advance_on", "days_per_turn"})


def load_cases(root: Path = DATA) -> tuple[dict, dict]:
    """load_cases（把所有組的所有變體讀進來）。
    回兩樣東西：

        tasks    {"COLO-01/v1_postponed": 任務定義}
        scripts  {"COLO-01/v1_postponed": {"reference": [...],
                                           "alternatives": [...]}}

    兩者分開回傳：腳本不是任務定義的一部分，
    沒有腳本的任務照樣載得起來。
    """
    tasks: dict = {}
    scripts: dict = {}
    folder = root / "cases"
    if not folder.is_dir():
        raise ValueError(f"找不到任務資料夾：{folder}")
    for case_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
        case = load_yaml(case_dir / "case.yaml")
        _check_case(case_dir, case)
        cid = case["case_id"]
        if cid != case_dir.name:
            raise ValueError(
                f"{case_dir.name}/case.yaml：case_id 寫 {cid!r}，"
                f"跟資料夾名字不一樣。改資料夾名的時候要一起改")
        pl = case_dir / "places.yaml"
        case["_places"] = load_yaml(pl) if pl.exists() else None
        found = 0
        for p in sorted(case_dir.glob("v*.yaml")):
            v = load_yaml(p)
            _check_variant(p, v)
            if v["variant"] != p.stem:
                raise ValueError(
                    f"{p}：variant 寫 {v['variant']!r}，但檔名是 {p.stem!r}。"
                    f"代號以欄位為準，但兩者刻意要求一致 —— "
                    f"不然改檔名會以為代號也改了")
            tid = f"{cid}/{v['variant']}"
            tasks[tid] = _merge(case, v, tid)
            found += 1
        if not found:
            raise ValueError(f"{case_dir.name}：一個變體都沒有（要有 v*.yaml）")
        if not any(_is_clean(t) for tid, t in tasks.items()
                   if t["case_id"] == cid):
            raise ValueError(
                f"{cid}：沒有任何變體宣告 from: empty。"
                f"每一組都要有一個「無擾動」的起點，其他變體才有東西可接")
        scripts.update(_load_scripts(case_dir, cid, tasks))
    return tasks, scripts


def _is_clean(task: dict) -> bool:
    """_is_clean（這個變體是不是「無擾動」的那一個）。
    判準是它宣告 from: empty，不是名字叫不叫 clean。
    """
    return (task.get("initial_state") or {}).get("from", "empty") == "empty"


def clean_variant(tasks: dict, case_id: str) -> str:
    """clean_variant（某一組的「無擾動」變體的代號）。
    其他變體的 from: v0_end 就是指它跑完之後的狀態。
    一組有兩個無擾動變體會報錯。
    """
    hits = [tid for tid, t in tasks.items()
            if t["case_id"] == case_id and _is_clean(t)]
    if not hits:
        raise ValueError(f"{case_id}：找不到無擾動的變體")
    if len(hits) > 1:
        raise ValueError(f"{case_id}：有兩個以上無擾動的變體 {hits}，起點不唯一")
    return hits[0]


def _merge(case: dict, variant: dict, tid: str) -> dict:
    """_merge（把整組共用的設定與這個變體的設定合起來）。
    變體寫了就用變體的，沒寫就用整組的。
    時鐘是逐欄合併，變體通常只覆寫起始日。
    """
    clock = {**(case.get("clock") or {}), **(variant.get("clock") or {})}
    bad = set(clock) - CLOCK_KEYS
    if bad:
        raise ValueError(f"{tid}：clock 有不認得的欄位 {sorted(bad)}")
    for key in ("start_date", "end_date", "advance_on", "days_per_turn"):
        if key not in clock:
            raise ValueError(f"{tid}：clock 缺少 {key}（case.yaml 或變體要寫）")
    if clock["advance_on"] not in ADVANCE_ON:
        raise ValueError(f"{tid}：不認得的 advance_on {clock['advance_on']!r}")
    if clock["start_date"] > clock["end_date"]:
        raise ValueError(f"{tid}：起始日晚於結束日")

    for ev in variant.get("event_script") or []:
        if ev["type"] not in EVENT_TYPES:
            raise ValueError(f"{tid}：不認得的事件型別 {ev['type']!r}")
        if not (clock["start_date"] <= ev["date"] <= clock["end_date"]):
            raise ValueError(f"{tid}：事件 {ev['date']} 落在時鐘區間外")

    # 設計紀錄，原樣帶著走。內容不檢查、不使用。
    manifest = copy.deepcopy(variant.get("manifest")
                             or case.get("manifest") or None)

    return {
        "task_id": tid,
        "case_id": case["case_id"],
        "variant": variant["variant"],
        # 標題兩層都可以有：整組一個、變體一個。印出來的時候接起來。
        "case_title": case.get("title", ""),
        "title": variant.get("title", ""),
        "patient_id": case["patient_id"],
        # 世界由這些元素組成。整組共用，所以五個變體的世界完全相同。
        "elements": list(case["elements"]),
        "distractors": list(case.get("distractors") or []),
        "start_node": variant.get("start_node") or case.get("start_node") or "C0",
        "initial_state": copy.deepcopy(variant.get("initial_state")
                                       or {"from": "empty"}),
        "clock": clock,
        "event_script": copy.deepcopy(variant.get("event_script") or []),
        "manifest": manifest,
        # 這個變體要往病人的個人資料補什麼（例如他自己先停了藥）。
        "patient_doc_patch": copy.deepcopy(
            variant.get("patient_doc_patch") or {}),
        # 情境自己的地點（病人住處與交通時間）
        "places": case.get("_places"),
    }


def _load_scripts(case_dir: Path, cid: str, tasks: dict) -> dict:
    """_load_scripts（讀這一組的腳本）。
      - 腳本是額外的檢查素材，不是任務定義的一部分，
        一個任務沒有腳本也是合法的任務
      - 腳本指到不存在的變體要報錯
    """
    out: dict = {}
    sdir = case_dir / "scripts"
    if not sdir.is_dir():
        return out
    for p in yaml_files(sdir):
        d = load_yaml(p)
        tid = f"{cid}/{d['variant']}"
        if tid not in tasks:
            raise ValueError(f"{p}：腳本指到不存在的變體 {d['variant']!r}")
        unknown = set(d) - {"variant", "reference", "alternatives"}
        if unknown:
            raise ValueError(f"{p}：腳本檔有不認得的欄位 {sorted(unknown)}")
        out[tid] = {
            # reference（標準走法）：這一題的一個標準答案。
            "reference": d.get("reference") or [],
            # alternatives（別的走法）：走到不同狀態的其他路徑。
            "alternatives": [dict(a) for a in (d.get("alternatives") or [])],
        }
    return out


def _check_case(case_dir: Path, case: dict) -> None:
    unknown = set(case) - CASE_KEYS
    if unknown:
        raise ValueError(f"{case_dir.name}/case.yaml：不認得的欄位 {sorted(unknown)}")
    for key in ("case_id", "patient_id"):
        if not case.get(key):
            raise ValueError(f"{case_dir.name}/case.yaml：缺少 {key}")
    if not case.get("elements"):
        raise ValueError(
            f"{case_dir.name}/case.yaml：缺少 elements。"
            f"世界是組裝出來的，每一組都要宣告它需要哪些元素")
    both = set(case["elements"]) & set(case.get("distractors") or [])
    if both:
        raise ValueError(
            f"{case_dir.name}/case.yaml：{sorted(both)} 同時被列為必要與干擾。"
            f"一個元素不可能既是答案又是干擾")


def _check_variant(path: Path, v: dict) -> None:
    unknown = set(v) - VARIANT_KEYS
    if unknown:
        raise ValueError(f"{path}：不認得的欄位 {sorted(unknown)}"
                         f"（腳本要放在 scripts/ 底下，不要寫在任務檔裡）")
    if not v.get("variant"):
        raise ValueError(f"{path}：缺少 variant")


def event_days(task: dict) -> list[str]:
    """event_days（這個任務會在哪幾天發生事件），已排序。"""
    return sorted({ev["date"] for ev in task.get("event_script") or []})


def world_elements(task: dict) -> list[str]:
    """world_elements（組出這一題的世界要用哪些元素）＝ 必要 ＋ 干擾。
    同一組的五個變體共用這份清單，所以它們的世界完全相同。
    """
    return sorted(set(task["elements"]) | set(task.get("distractors") or []))


def cases_of(tasks: dict) -> dict:
    """cases_of（把變體依組分群），回 {組代號: [變體代號…]}，都已排序。"""
    out: dict = {}
    for tid in sorted(tasks):
        out.setdefault(tasks[tid]["case_id"], []).append(tid)
    return out
