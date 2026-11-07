"""把一次執行的結果寫成檔案。
  runs/<這次執行的編號>/
  ├ config.json      這次用的設定（哪個供應商、哪個模型、亂數種子、溫度）
  ├ state.json       最終狀態的原樣傾印
  ├ episode.json     最終狀態 ＋ 完整變更紀錄 ＋ 呼叫紀錄，
  │                      一個檔案就看得懂整件事。給分析與公開釋出用
  └ transcript.jsonl 跟模型來回的對話逐字稿（照腳本跑的沒有這個）

  - 寫成檔案是為了之後能離線重算：state.json 是完整的最終狀態，
    change_log 是過程中每一次改動，兩個加起來足以在完全不碰模型的
    情況下重建整件事
  - 這裡不寫任何對錯判定，只寫實際發生的事
"""
from __future__ import annotations

import json
from pathlib import Path

RUNS = Path("runs")


def run_id(task_id: str, agent: str, model: str | None, i: int = 1) -> str:
    """run_id（這次執行的目錄名）。看得懂、排得了序、不含時間戳。
    不用時間戳：同一組設定跑兩次會覆寫同一個目錄。
    要保留多次就用 --repeat，第幾次會寫進目錄名。
    """
    # 任務代號含斜線（COLO-01/v1_postponed），不能直接當目錄名，
    # 那會變成兩層目錄。
    parts = [task_id.replace("/", "_"), agent]
    if model:
        parts.append(model.replace(":", "-").replace("/", "-"))
    if i > 1:
        parts.append(f"r{i}")
    return "_".join(parts)


def episode(rid: str, config: dict, session) -> dict:
    """episode（一次執行的完整紀錄），一個檔案就看得懂整件事。
      - state.json 是最終狀態的原樣傾印；這個檔是這一次到底發生了
        什麼：哪一天改了什麼、改之前是什麼、改之後是什麼
      - 裡面不含地面真值與指引
    """
    snap = session.snapshot()
    st = snap["state"]
    return {
        "run_id": rid,
        "config": config,
        "clock": {"start": session.task["clock"]["start_date"],
                  "end": session.task["clock"]["end_date"],
                  "final_day": snap["today"]},
        # 變更紀錄：每一次狀態改變的「改之前 / 改之後」。
        "change_log": st.get("change_log") or [],
        "tool_calls": st.get("tool_call_log") or [],
        "final_state": {k: v for k, v in st.items()
                        if k not in ("change_log", "tool_call_log")},
        "records": snap["records"],
        "slots": snap["slots"],
    }


def write(rid: str, config: dict, session,
          transcript: list | None = None, root: Path = RUNS) -> Path:
    """write（把上面那幾個檔案全部寫出去），回傳寫到哪個目錄。"""
    out = root / rid
    out.mkdir(parents=True, exist_ok=True)
    _json(out / "config.json", config)
    _json(out / "state.json", session.snapshot())
    _json(out / "episode.json", episode(rid, config, session))
    if transcript:
        with open(out / "transcript.jsonl", "w", encoding="utf-8") as f:
            for row in transcript:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    return out


def read_state(rid: str, root: Path = RUNS) -> dict:
    """read_state（把某一次執行的最終狀態讀回來）。
    離線重算的入口：不必重跑模型，直接拿存下來的最終狀態算。
    """
    with open(root / rid / "state.json", encoding="utf-8") as f:
        return json.load(f)


def _json(path: Path, obj) -> None:
    """_json（寫一個 JSON 檔）。欄位照字母排序，所以兩次執行的檔案逐字相同。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str, sort_keys=True)
