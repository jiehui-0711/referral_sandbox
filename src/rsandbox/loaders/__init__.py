"""資料載入層：把 data/ 底下的 YAML 讀成程式用的資料結構。
一個資料分支一個載入器：

    world.py      組裝世界：固定層 ＋ 任務選用的元素
    guidelines.py 條件式指引。Agent 唯一讀得到的臨床知識，
                  跟著元素走，用 search_guidelines 檢索
    records.py    病歷              每個 episode 一份可寫的副本
    patients.py   病人的個人視角資料
    graph.py      節點樹與工具表    從幾份單一事實來源推導出來
    cases.py      任務：一個資料夾一組，一組底下幾個變體
    state.py      執行期資料表的格式定義（schema）
    prompts.py    被測 Agent 的系統提示詞
      元素的 guidelines/   條件式指引　Agent 用 search_guidelines 檢索
      truth/               用藥原因的分類對照表　沒有任何工具查得到
"""
from __future__ import annotations

from pathlib import Path

from ..paths import DATA
from . import (cases, graph, guidelines, patients, prompts, records, state,
               world)


def load_all(root: Path = DATA) -> dict:
    """load_all（把所有靜態資料載成一包）。
    「靜態」是指跑之前就決定、跑的時候不會變的東西。
    執行期會變的狀態是另一回事，由 state.empty_state() 給一份空的。
    """
    elements = world.load_elements(root)
    tasks_, scripts_ = cases.load_cases(root)

    # 一組任務一份世界。同一組的五個變體共用，因為它們的元素清單相同。
    # 用元素清單當快取鍵：清單一樣的組共用同一份世界，不重複組裝。
    worlds: dict = {}
    by_key: dict = {}
    for tid, t in tasks_.items():
        key = (tuple(cases.world_elements(t)), id(t.get("places")))
        if key not in by_key:
            by_key[key] = world.build_world(root, cases.world_elements(t),
                                            case_places=t.get("places"),
                                            elements=elements)
        worlds[t["case_id"]] = by_key[key]

    # 工具表只建一份，而且參數的合法值涵蓋所有元素可能出現的科別與
    # 診次型別。工具是固定層的東西，不隨任務改變。
    all_world = world.build_world(root, None, elements=elements)
    world_enums = world.enum_values(all_world)
    tools = graph.load_tools(root, world_enums)
    gates = graph.load_gates(root, tools)

    # 節點樹跟著世界走：只選了大腸鏡元素的那一題，骨科節點
    # 根本不存在，走過去會回「查無此科」。節點是從機構與科別推導的，
    # 所以世界一組好，節點樹就自動對了。
    node_trees: dict = {}
    for cid, w in worlds.items():
        nodes = graph.build_nodes(w["institutions"], w["departments"])
        graph.attach_tools(nodes, tools, gates)
        node_trees[cid] = nodes
    return {
        # 每一組任務自己的世界。沒有「全域的世界」這種東西了。
        "worlds": worlds,
        "elements": elements,
        "guidelines": guidelines.load_guidelines(root),
        "records": records.fresh_copy(root),
        # 病人的個人視角資料。病人 Agent 只讀這個，讀不到病歷。
        "patient_docs": patients.load_patient_docs(root),
        # 病人 Agent 的行為規則。所有情境共用一份。
        "patient_instructions": prompts.prompt_text("patient_agent", root),
        "node_trees": node_trees,
        "tools": tools,
        # 任務與腳本分開回傳：腳本是額外的檢查素材，不是任務定義
        # 的一部分。載入任務的路徑上完全不依賴腳本。
        "tasks": tasks_,
        "scripts": scripts_,
        "schema": state.load_schema(root),
    }


__all__ = ["load_all", "world", "guidelines", "records", "patients",
           "graph", "cases", "state", "prompts"]
