"""載入節點樹與工具表（graph/）。
「節點」是 Agent 可以站的地方：自己的診所、每一間醫院、每一科。
Agent 站在哪裡，決定它能用哪些工具。

節點由世界裡的機構與科別自動產生。
  - 工具參數的合法值：從指引或世界資料複製
  - 每個節點有哪些工具：從工具所在的資料夾（＝權限層級）反推
  - 節點本身與相鄰關係：從機構、科別生成
  - 每支工具都會權限檢查：自動補上，工具檔不手寫
  - 每支工具屬於哪一類關卡：看它放在 data/tools/ 的哪個資料夾
相鄰關係：
  - 診所與各家醫院：彼此全連通
  - 科別節點只連到自己所屬的醫院
  - 走路的成本只有步數，而步數是充足的。
"""
from __future__ import annotations

import copy
from pathlib import Path

from ..paths import DATA, load_yaml, yaml_files

# PERMISSION_PRECONDITION：自動補到每支工具上的權限檢查。
# 工具檔不手寫這一條，載入器會補。
# 目前只有一個被測角色，所以這一關永遠通過。
PERMISSION_PRECONDITION = {
    "type": "permission",
    "rule": "caller.role in tool.permissions",
    "on_fail": "permission_denied",
}

# ALLOWED_PRECONDITION_TYPES：前置條件只能是這兩種，
# 沒有 clinical（臨床）這一種。
ALLOWED_PRECONDITION_TYPES = (
    "permission",   # 權限／位置
    "admin",        # 行政（號滿了、日期過了、少一張轉診單）
)


# 工具資料夾名 → 權限層級。四個資料夾全部映到 any：工具可以從任何節點
# 呼叫，位置不是任何事情的前提。
# 資料夾名字留著當資訊層級的分組（通用／診所／機構層／科別層），
# 那是「這支工具讀世界的哪一層」，跟「你要站在哪裡」是兩件事。
#   - 轉診協調員坐在自己診所的桌子前工作，線上查號、線上訂號
#   - 被測的負擔是病人要跑幾趟，不是協調員要跑幾趟
LEVEL_DIRS = {"global": "any", "origin": "any",
              "institution": "any", "department": "any"}


def load_gates(root: Path = DATA, tools: dict | None = None) -> dict:
    """load_gates（每一種節點可以用哪些工具），回傳 {節點種類: [工具]}。
    這份對應不是另外登記的，是從資料夾推導的：

        data/tools/global/       → 任何節點
        data/tools/origin/       → 轉出端診所
        data/tools/institution/  → 醫院大門
        data/tools/department/   → 科別節點
        data/elements/<元素>/tools/  → 該元素專屬，層級寫在工具自己的
                                       level 欄位
    存放位置就是答案，不另外登記一份清單。

    擋下來的訊息固定是「這個動作在你目前的位置不能做」，不提示該去哪裡。
    """
    tools = tools if tools is not None else load_tools(root)
    out: dict = {}
    for t in tools.values():
        out.setdefault(t["level"], []).append(t["tool_id"])
    for k in out:
        out[k].sort()
    return out


def build_nodes(institutions: dict, departments: dict) -> dict:
    """build_nodes（從機構與科別生成整棵節點樹），不另外寫一個檔。
    節點分四種：
        root        最上層的沙盒本身
        origin      Agent 任職的診所（不可被轉介）
        hospital    可以轉診過去的醫院
        department  某一間醫院的某一科

    「相鄰」與「父子」是兩件不同的事：
    父子（parent）    是結構 —— 這一科屬於哪家醫院
    相鄰（neighbours）是移動 —— 從這裡可以一步走到哪裡
    移動只看相鄰，不看父子。
    """
    nodes: dict = {
        "SANDBOX": {"node_id": "SANDBOX", "type": "root", "parent": None,
                    "name": "Referral sandbox", "name_zh": "轉診沙盒"},
    }
    tops = []
    for iid, inst in sorted(institutions.items()):
        kind = "origin" if inst.get("role") == "origin" else "hospital"
        nodes[iid] = {"node_id": iid, "type": kind, "parent": "SANDBOX",
                      "name": inst.get("name"), "name_zh": inst.get("name_zh"),
                      "institution_id": iid}
        tops.append(iid)
    for did, dep in sorted(departments.items()):
        nodes[did] = {"node_id": did, "type": "department",
                      "parent": dep["institution_id"],
                      "name": dep.get("name"), "name_zh": dep.get("name_zh"),
                      "institution_id": dep["institution_id"],
                      "dept": dep["dept"]}

    for n in nodes.values():
        n["edges"] = []
        n["tools"] = []
        n["neighbours"] = []
    for nid, n in sorted(nodes.items()):
        par = n.get("parent")
        if par is not None:
            nodes[par]["edges"].append({"to": nid, "rel": "contains"})
    # 機構之間全連通：想去哪一家都一步到得了。
    for a in tops:
        nodes[a]["neighbours"] = sorted(x for x in tops if x != a)
    # 科別節點可以回到任何一間機構，但要進別家的科，得先進那家醫院。
    # 科與科之間不直接相鄰。
    for did, n in sorted(nodes.items()):
        if n["type"] != "department":
            continue
        n["neighbours"] = list(tops)
        nodes[n["parent"]]["neighbours"].append(did)
    for n in nodes.values():
        n["neighbours"] = sorted(set(n["neighbours"]))
    return nodes


def _resolve_enums(tool: dict, world_enums: dict) -> None:
    """_resolve_enums（把「去哪裡拿合法值」換成實際的合法值清單）。
    工具的 YAML 裡寫的是 enum_from: world.某個名稱，意思是這個參數的
    合法值去世界資料拿（機構清單、科別清單、診次型別）。
      - 世界資料是合法值的唯一來源，工具不自己另寫一份清單
      - 同時寫 enum 與 enum_from 會報錯
      - search_guidelines 的查詢詞是自由文字，沒有 enum
    """
    for name, spec in (tool.get("params") or {}).items():
        ref = spec.pop("enum_from", None)
        if ref is None:
            continue
        if "enum" in spec:
            raise ValueError(f"{tool['tool_id']}.{name}：enum 與 enum_from 只能有一個")
        if not ref.startswith("world."):
            raise ValueError(
                f"{tool['tool_id']}.{name}：enum_from 只能是 world.*，"
                f"收到 {ref!r}")
        key = ref.split(".", 1)[1]
        if key not in world_enums:
            raise ValueError(f"{tool['tool_id']}.{name}：不認得的 enum_from {ref!r}")
        spec["enum"] = list(world_enums[key])


def load_tools(root: Path = DATA, world_enums: dict | None = None,
               include_masked: bool = False) -> dict:
    """load_tools（載入工具表），回傳 {工具名: 工具定義}。
      - include_masked=False：跳過 masked: true 的工具，沙盒本身用這個
      - include_masked=True：連被屏蔽的一起回傳，定義上帶著 masked
    """
    if world_enums is None:
        # 參數的合法值要涵蓋所有元素可能出現的科別與診次型別。
        # 工具表是固定層的東西，不因某一題選了哪些元素而改變。
        from .world import build_world, enum_values
        world_enums = enum_values(build_world(root, None))

    tools: dict = {}
    sources = []
    for folder, level in sorted(LEVEL_DIRS.items()):
        d = root / "tools" / folder
        if d.is_dir():
            sources += [(p, level) for p in yaml_files(d)]
    # 元素專屬／覆寫的工具。層級寫在工具自己的 level 欄位。
    el_dir = root / "elements"
    if el_dir.is_dir():
        for e in sorted(p for p in el_dir.iterdir() if p.is_dir()):
            if (e / "tools").is_dir():
                sources += [(p, None) for p in yaml_files(e / "tools")]
    for p, level in sources:
        for t in (load_yaml(p) or {}).get("tools") or []:
            t = dict(t)
            t["level"] = t.get("level") or level
            if t["level"] not in set(LEVEL_DIRS.values()):
                raise ValueError(
                    f"{t.get('tool_id')}：不認得的權限層級 {t['level']!r}。"
                    f"元素專屬的工具要自己寫 level")
            t = copy.deepcopy(t)
            tid = t["tool_id"]
            if tid in tools:
                raise ValueError(f"tool {tid!r} 定義了兩次")
            if t.get("masked") and not include_masked:
                # masked: true 的工具不進工具表，所以：
                #   - 不在提示詞的工具清單裡、不在原生 schema 裡
                #   - 呼叫它等於呼叫一支不存在的工具
                #   - 定義與 handler 都留著，刪掉那一行就恢復
                continue
            _resolve_enums(t, world_enums)
            pre = list(t.get("precondition") or [])
            for c in pre:
                if c.get("type") not in ALLOWED_PRECONDITION_TYPES:
                    raise ValueError(
                        f"{tid}：precondition.type {c.get('type')!r} 不合法。"
                        f"只能是 {ALLOWED_PRECONDITION_TYPES} —— "
                        f"沒有 clinical（臨床）這一種")
                if c["type"] == "permission":
                    raise ValueError(f"{tid}：permission 前置條件由載入器自動補，不要手寫")
            t["precondition"] = [dict(PERMISSION_PRECONDITION)] + pre
            t["_file"] = p.name
            tools[tid] = t
    return tools


def attach_tools(nodes: dict, tools: dict, gates: dict) -> None:
    """attach_tools（把工具掛到節點上，並把關卡類別寫回工具）。
    每個節點能用的工具 ＝ 任何地方都能用那一組 ＋ 這種節點能用那一組，
    所以不必逐個節點維護一份工具清單。
    每一支工具恰好屬於一組，因為它的層級是從它所在的資料夾推導的。
    """
    where: dict[str, str] = {}
    for kind, tids in sorted(gates.items()):
        for tid in tids:
            if tid in where:
                raise ValueError(f"tool {tid!r} 屬於兩個權限層級："
                                 f"{where[tid]} 與 {kind}")
            if tid not in tools:
                raise ValueError(f"權限層級裡列了不存在的工具 {tid!r}")
            where[tid] = kind
    missing = sorted(set(tools) - set(where))
    if missing:
        raise ValueError(f"這些工具沒有權限層級（放錯資料夾？）：{missing}")

    for tid, kind in sorted(where.items()):
        tools[tid]["gate"] = kind
    for nid, n in sorted(nodes.items()):
        allowed = list(gates.get("any") or [])
        allowed += list(gates.get(n["type"]) or [])
        n["tools"] = sorted(set(allowed))
        tools_here = n["tools"]
        for tid in tools_here:
            tools[tid].setdefault("nodes", [])
    for tid, t in sorted(tools.items()):
        t["nodes"] = sorted(nid for nid, n in nodes.items() if tid in n["tools"])


def build_graph(root: Path = DATA, world_enums: dict | None = None,
                institutions: dict | None = None,
                departments: dict | None = None) -> tuple[dict, dict]:
    """build_graph（把節點樹與工具表組好），回傳 (節點樹, 工具表)。"""
    if institutions is None or departments is None:
        from .world import load_departments, load_institutions
        institutions = load_institutions(root)
        departments = load_departments(institutions)
    nodes = build_nodes(institutions, departments)
    tools = load_tools(root, world_enums)
    attach_tools(nodes, tools, load_gates(root))
    return nodes, tools
