"""組裝世界：固定層 ＋ 任務選用的元素。
世界是組裝出來的，不是全域共用的：
    data/fixed/        固定層。每一個任務都有：C0 與 H1–H4、
                       全院通行的規定、機構的地點
    data/elements/     元素庫。全域存放，每個元素一個資料夾：
                       科別、項目、科別層與項目層的規定、診次規則
    data/cases/<組>/   情境自己的資料：病人、病歷、住處與交通
任務清單用 elements 選必要元素，distractors 選干擾元素。

  - 世界組裝之後就不再改變，只有診次的已訂人數會被工具動到
  - Agent 只能透過工具讀世界，而且只能讀它走到的那個節點底下的東西
  - 三層結構（機構 → 科別 → 項目），隱含的負擔可以藏在任何一層
"""
from __future__ import annotations

import copy
import datetime as _dt
from pathlib import Path

from ..paths import DATA, load_yaml, yaml_files

WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


#  機構 / 科別 / 項目

def load_institutions(root: Path = DATA) -> dict:
    """load_institutions（載入機構），回傳 {機構代號: 機構定義}。
    機構屬固定層：每一個任務都有這五家。科別與項目不在這裡，
    它們在元素庫，由任務清單選用。

    讀兩個資料夾：
        fixed/origin/        Agent 自己任職的那一家（不可被轉介）
        fixed/institutions/  可以轉診過去的那幾家
    """
    out: dict = {}
    for folder in ("origin", "institutions"):
        for p in yaml_files(root / "fixed" / folder):
            d = load_yaml(p)
            iid = d["institution_id"]
            if iid in out:
                raise ValueError(f"institution {iid!r} 定義了兩次")
            out[iid] = d
    return out


def load_departments(institutions: dict) -> dict:
    """load_departments（載入科別），回傳 {科別代號: 科別定義}。
    從各機構的科別清單反推出來，不另外維護一份名單。

    「這一科不存在」與「這一科存在但沒公布資訊」是兩件不同的事：
    不存在      -> 這份清單裡沒有那一筆，工具回「查無此科」
    沒公布資訊  -> 狀態標成 empty，走得進去，查詢回「沒有公布」
    """
    out: dict = {}
    for iid, inst in sorted(institutions.items()):
        for dep in inst.get("departments") or []:
            d = dict(dep)
            d["institution_id"] = iid
            d.setdefault("status", "published")
            d.setdefault("services", [])
            d.setdefault("block_ids", [])
            d.setdefault("dept", d["dept_id"].split("_", 1)[-1])
            if d["dept_id"] in out:
                raise ValueError(f"department {d['dept_id']!r} 定義了兩次")
            out[d["dept_id"]] = d
    return out


def load_services(departments: dict) -> dict:
    """load_services（載入服務項目）：三層結構的第三層。
    回傳 {(科別代號, 項目代號): 項目定義} —— 用兩個代號一組當鍵，
    因為不同科可能有同名的項目。
    """
    out: dict = {}
    for did, dep in sorted(departments.items()):
        for svc in dep.get("services") or []:
            s = dict(svc)
            s["dept_id"] = did
            s["institution_id"] = dep["institution_id"]
            s["dept"] = dep["dept"]
            s.setdefault("block_ids", [])
            s.setdefault("prep_block_ids", [])
            s.setdefault("is_sedated", False)
            s.setdefault("stay_nights", None)
            key = (did, s["code"])
            if key in out:
                raise ValueError(f"service {key} 定義了兩次")
            out[key] = s
    return out


def slot_types(services: dict) -> list[str]:
    """slot_types（所有項目用到的診次型別，取聯集並排序）。"""
    return sorted({s["slot_type"] for s in services.values() if s.get("slot_type")})


def enum_values(world: dict) -> dict:
    """enum_values（工具參數可以從世界資料拿的合法值清單），回 {名稱: 清單}。
    工具的 YAML 寫 enum_from: world.<名稱>，名稱就是這裡的鍵。
    世界資料是這些清單的唯一來源，工具不自己另寫一份。

    科別與項目的代號刻意不在這裡：它們要從三層資訊逐層查出來
    （get_institution_info 回該院有哪些科，get_department_info 回該科有
    哪些項目）。把它們攤成全域清單等於抄掉那條發現路徑。
    """
    return {
        "institutions": sorted(world["institutions"]),
        "slot_types": world["slot_types"],
    }


#  資訊段落

def _norm_blocks(rows, out: dict) -> dict:
    """_norm_blocks（把一批段落正規化後放進 out）。"""
    for b in rows or []:
        d = dict(b)
        d.setdefault("reveal", "always")
        d.setdefault("role", "task")
        d["visible"] = d["reveal"] == "always"
        if d["id"] in out:
            raise ValueError(f"info_block {d['id']!r} 定義了兩次")
        out[d["id"]] = d
    return out


def load_blocks(root: Path = DATA) -> dict:
    """load_blocks（載入固定層的資訊段落），回傳 {段落代號: 段落}。
    只有機構層的段落在這裡。科別層與項目層的段落跟著元素走，
    由 build_world 在組裝的時候加進來。
    Agent 讀得到的那些英文公告原文就在這裡。
      - 段落上標註的角色（本題的負擔還是干擾）與揭露時機是給人看的設定，
        絕對不會出現在工具回傳裡
      - 可見（visible）是執行期欄位：標成等轉診被接受才揭露的段落
        一開始不可見，送出轉診成功之後才翻成可見
    """
    data = load_yaml(root / "fixed" / "blocks.yaml") or {}
    return _norm_blocks(data.get("blocks"), {})


#  診次：規則 → 逐筆

def _dates(date_from: str, date_to: str, weekdays: list[str]):
    """_dates（一段日期裡，星期幾符合條件的每一天）。完全確定，沒有隨機。"""
    want = {WEEKDAYS.index(w) for w in weekdays}
    cur = _dt.date.fromisoformat(date_from)
    end = _dt.date.fromisoformat(date_to)
    while cur <= end:
        if cur.weekday() in want:
            yield cur.isoformat()
        cur += _dt.timedelta(days=1)


def expand_slots(rules: list, services: dict) -> dict:
    """expand_slots（把排程規則展開成一筆一筆的診次）。
    回傳 {診次代號: 診次}。

      - 展開的結果完全確定：日期由起、迄、星期幾三個欄位決定，
        順序由診次代號決定。同一份資料展開兩次，結果逐字相同
      - 規則跟著元素走，不是一個全域大檔
    """
    out: dict = {}
    for rule in rules or []:
        key = (f"{rule['institution_id']}_{rule['dept']}", rule["service"])
        svc = services.get(key)
        if svc is None:
            raise ValueError(f"{rule['rule_id']}：{key} 這個項目不存在")
        exc = {e["date"]: e for e in rule.get("exceptions") or []}
        pre = {p["date"]: p["count"] for p in rule.get("prebooked") or []}
        for day in _dates(rule["date_from"], rule["date_to"], rule["weekdays"]):
            cap = rule["capacity"]
            if day in exc:
                cap = exc[day].get("capacity", cap)
            sid = (f"{rule['institution_id']}-{rule['dept']}-{rule['service']}"
                   f"-{day.replace('-', '')}-{rule['period']}")
            if sid in out:
                raise ValueError(f"slot {sid!r} 產生了兩次")
            out[sid] = {
                "slot_id": sid,
                "rule_id": rule["rule_id"],
                "institution_id": rule["institution_id"],
                "dept_id": f"{rule['institution_id']}_{rule['dept']}",
                "dept": rule["dept"],
                "service": rule["service"],
                "slot_type": svc["slot_type"],
                "date": day,
                "period": rule["period"],
                "capacity": cap,
                "booked_count": min(pre.get(day, 0), cap),
                # 狀態是執行期欄位。容量被例外條款設成 0 的那一天標成
                # 停診，Agent 查號源時查不到它。
                "status": "cancelled" if cap == 0 else "open",
            }
    return out


#  地點與交通

def load_places(root: Path = DATA) -> tuple[dict, dict]:
    """load_places（載入地點與交通時間）。
    回傳兩樣東西：{地點代號: 地點} 與 {(起點, 終點): 分鐘}。

    交通時間是單程分鐘，來回視為相同，載入時自動補上反方向，
    所以資料檔只要寫一次。
    """
    return merge_places([load_yaml(root / "fixed" / "places.yaml") or {}])


def merge_places(sources: list) -> tuple[dict, dict]:
    """merge_places（把好幾份地點資料併成一份）。
    地點有三個來源：固定層（機構）、元素、情境（病人住處）。
    併的時候同一個地點代號出現兩次就報錯。
    """
    places: dict = {}
    travel: dict = {}
    for data in sources:
        for p in (data or {}).get("places") or []:
            if p["id"] in places:
                raise ValueError(f"地點 {p['id']!r} 定義了兩次")
            places[p["id"]] = p
        for t in (data or {}).get("travel") or []:
            for a, b in ((t["from"], t["to"]), (t["to"], t["from"])):
                travel[(a, b)] = t["minutes"]
    for pid in places:
        travel.setdefault((pid, pid), 0)
    return places, travel


def load_elements(root: Path = DATA) -> dict:
    """load_elements（把元素庫整個讀進來），回傳 {元素代號: 元素}。
      - 元素庫是全域的：所有元素的內容與編號在每個任務裡都一樣，
        任務清單只決定這一題用哪幾個，不改內容
      - 一個元素 = 一個臨床能力包，跨醫院。它宣告：
            departments   需要哪些科別與項目（掛在既有醫院之下）
            blocks        那些科別／項目的規定（科別層與項目層）
            slot_rules    那些項目的診次規則
        另外可以有自己的 places.yaml
    """
    out: dict = {}
    folder = root / "elements"
    if not folder.is_dir():
        raise ValueError(f"找不到元素庫：{folder}")
    for d in sorted(p for p in folder.iterdir() if p.is_dir()):
        el = load_yaml(d / "element.yaml")
        eid = el["element_id"]
        if eid != d.name:
            raise ValueError(f"{d.name}/element.yaml：element_id 寫 {eid!r}，"
                             f"跟資料夾名字不一樣")
        if eid in out:
            raise ValueError(f"元素 {eid!r} 定義了兩次")
        pl = d / "places.yaml"
        el["places_file"] = load_yaml(pl) if pl.exists() else None
        out[eid] = el
    return out


def merge_departments(institutions: dict, elements: dict, chosen) -> None:
    """merge_departments（把選用元素的科別掛到機構上）。就地修改 institutions。
      - 同一科可以被多個元素宣告（骨科同時被人工膝關節與骨折用到），
        這裡合併，屬性不一致就報錯
      - 項目用 (科別, 項目代號) 當鍵，重複定義一樣報錯
    """
    for eid in chosen:
        for dep in elements[eid].get("departments") or []:
            did = dep["dept_id"]
            iid = did.split("_", 1)[0]
            inst = institutions.get(iid)
            if inst is None:
                raise ValueError(f"元素 {eid} 的科別 {did} 指到不存在的機構 {iid}")
            existing = next((d for d in inst.setdefault("departments", [])
                             if d["dept_id"] == did), None)
            if existing is None:
                inst["departments"].append(copy.deepcopy(dep))
                continue
            # 已經有人宣告過這一科：核心屬性必須一致
            for key in ("dept", "name", "status", "block_ids"):
                if dep.get(key) != existing.get(key):
                    raise ValueError(
                        f"科別 {did} 被兩個元素宣告，但 {key} 不一致："
                        f"{existing.get(key)!r} vs {dep.get(key)!r}（元素 {eid}）")
            have = {s["code"] for s in existing.get("services") or []}
            for svc in dep.get("services") or []:
                if svc["code"] in have:
                    raise ValueError(f"項目 ({did}, {svc['code']}) 被宣告了兩次")
                existing.setdefault("services", []).append(copy.deepcopy(svc))

    # 項目的順序照代號排，不照元素的載入順序，這樣 get_department_info
    # 回給 Agent 的清單順序才不會隨這一題選了哪些元素而變。
    for inst in institutions.values():
        for dep in inst.get("departments") or []:
            dep.setdefault("services", []).sort(key=lambda s: s["code"])
        dep_list = inst.get("departments") or []
        dep_list.sort(key=lambda d: d["dept_id"])


def build_world(root: Path = DATA, element_ids=None,
                case_places=None, elements: dict | None = None) -> dict:
    """build_world（組裝一份世界）：固定層 ＋ 選用的元素 ＋ 情境的地點。
        element_ids  這一題要用哪些元素（必要 ＋ 干擾）。None = 全部
        case_places  情境自己的地點資料（病人住處與交通時間）
        elements     元素庫。None 就自己載入

    元素清單裡有不存在的元素就直接報錯，不回傳半套的世界。
    """
    from .guidelines import load_guidelines
    elements = elements if elements is not None else load_elements(root)
    chosen = sorted(elements) if element_ids is None else sorted(set(element_ids))
    unknown = [e for e in chosen if e not in elements]
    if unknown:
        raise ValueError(f"元素庫裡沒有這些元素：{unknown}")

    institutions = copy.deepcopy(load_institutions(root))
    merge_departments(institutions, elements, chosen)

    blocks = load_blocks(root)
    for eid in chosen:
        _norm_blocks(elements[eid].get("blocks"), blocks)

    departments = load_departments(institutions)
    services = load_services(departments)

    rules = []
    for eid in chosen:
        rules += elements[eid].get("slot_rules") or []
    slots = expand_slots(rules, services)

    places, travel = merge_places(
        [load_yaml(root / "fixed" / "places.yaml") or {}]
        + [elements[eid].get("places_file") for eid in chosen]
        + ([case_places] if case_places else []))

    # 指引跟著元素走：這一題讀得到哪些指引，由元素清單決定。
    # 別題的必要指引就是這一題的干擾指引。
    all_g = load_guidelines(root)
    guidelines = {gid: g for gid, g in all_g.items() if g["element"] in set(chosen)}

    return {
        "institutions": institutions,
        "departments": departments,
        "services": services,
        "blocks": blocks,
        "slot_types": slot_types(services),
        "slots": copy.deepcopy(slots),
        "places": places,
        "travel": travel,
        "guidelines": guidelines,
        # 這一份世界是用哪些元素組出來的。
        "_elements": chosen,
    }
