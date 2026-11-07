"""Session（一次執行的全部狀態）。
一個 Session 綁住一個任務。「一次執行」＝一個案例從第一天跑到最後一天。
它身上掛的東西：

    world      世界：機構、科別、服務項目、診次。
               只有 slots 的 booked_count（已訂人數）會變，其餘不動
    world['guidelines']  條件式指引。Agent 靠 search_guidelines 檢索
    records    病歷。每個 episode 一份可寫的副本
    patient_doc 病人的個人視角資料。病人 Agent 只讀這個
    nodes      節點樹：Agent 可以站在哪些地方、彼此相不相鄰
    tools      工具表（唯讀）
    state      執行期的資料表。跑完之後交出來的就是這一包
    today      模擬時鐘。只有 end_turn（結束回合）會推進它

地面真值（truth/）不在這裡，Agent 的任何工具都碰不到。

Session.call() 是唯一會改變世界的入口。它永遠回傳信封、永遠寫一筆
呼叫紀錄，失敗不丟例外。
"""
from __future__ import annotations

import copy
import importlib

from ..loaders.state import empty_state
from . import changelog
from .envelope import ToolError, fail, ok
from .executor import check_gate, check_preconditions, validate_args
from .ids import IdMaker

TESTED_ROLE = "referral_agent"


class Ctx:
    """Ctx（context，上下文）：傳給 handler 的那一包東西。
    handler 只能透過它碰世界：它身上有什麼，handler 就只能動什麼。
    身上沒有地面真值的入口。
    """

    def __init__(self, session: "Session", tool: dict, args: dict) -> None:
        self.session = session
        self.tool = tool
        self.args = args
        self.world = session.world
        self.records = session.records
        self.state = session.state
        self.today = session.today
        self.role = session.role
        self.ids = session.ids
        # 這一案的病人。所有跟病人有關的工具都隱含作用在他身上。
        self.patient_id = session.task["patient_id"]
        # 病人的個人視角資料與演它的那個模型。病人讀不到病歷。
        self.patient_doc = session.patient_doc
        self.patient_backend = session.patient_backend
        self.patient_instructions = session.patient_instructions

    def new_id(self, table: str) -> str:
        """跟某張表要一個新編號（例如 appointments 要到 APT-003）。"""
        return self.ids.for_table(table)


class Session:
    def __init__(self, data: dict, task: dict, state: dict | None = None,
                 today: str | None = None, config: dict | None = None,
                 patient_backend=None) -> None:
        # 世界是組裝出來的，一組任務一份。只選了大腸鏡元素的那一題，
        # 骨科節點根本不存在。
        cid = task["case_id"]
        self.world = copy.deepcopy(data["worlds"][cid])
        # 指引全文：Agent 唯一讀得到的臨床知識。唯讀，不複製。
        self.records = copy.deepcopy(data["records"])
        # 病人的個人視角資料。每個 episode 一份可寫的副本，
        # 因為事件會往它的 log 追加。
        self.patient_doc = copy.deepcopy(
            data["patient_docs"][task["patient_id"]])
        self.patient_instructions = data["patient_instructions"]
        self.patient_backend = patient_backend
        self.nodes = data["node_trees"][cid]
        self.tools = data["tools"]
        self.task = task
        self.config = dict(config or {})
        self.role = TESTED_ROLE
        self.state = copy.deepcopy(state) if state else empty_state()
        self.today = today or task["clock"]["start_date"]
        self.state["sim_clock"]["today"] = self.today
        # Agent 每天從自己的診所開始。這件事寫在系統提示詞裡。
        self.start_node = task.get("start_node", "C0")
        self.state.setdefault("position", {"node": None})
        self.state["position"]["node"] = self.start_node
        self.ids = IdMaker()
        self.ids.seed_from(self.state)
        self._handlers: dict[str, object] = {}
        # 序號從繼承來的紀錄接下去，不從 0 重來：v1～v5 的起始狀態是
        # v0 跑完的狀態，裡面已經有一串序號了。
        self._seq = max([r.get("seq", 0) for r in self.state["tool_call_log"]]
                        + [r.get("seq", 0) for r in self.state["change_log"]]
                        + [0])

    # ---------------- 節點歸屬 ----------------

    def node_of(self, tool_id: str, args: dict) -> str | None:
        """node_of（這次呼叫發生在哪個節點），寫進呼叫紀錄。
        有些工具掛在很多個節點上（查號源、訂號、改期都是），
        光看工具名不知道它動的是哪一科，要從參數反推。
        """
        t = self.tools.get(tool_id)
        if t is None:
            return None
        nodes = t["nodes"]
        if len(nodes) == 1:
            return nodes[0]
        dept = args.get("dept_id")
        if dept in nodes:
            return dept
        sid = args.get("slot_id") or args.get("new_slot_id")
        slot = self.world["slots"].get(sid or "")
        if slot and slot["dept_id"] in nodes:
            return slot["dept_id"]
        return None

    # ---------------- 呼叫工具 ----------------

    def call(self, tool_id: str, args: dict | None = None) -> dict:
        """執行一次工具呼叫。永遠回傳信封，永遠寫一筆呼叫紀錄。
            {"ok": True,  "reason": None,   "data": {...}}   成功
            {"ok": False, "reason": "...",  "data": None}    失敗

        四道關卡，沒有第五道：
          - 第 1 關：工具存在
          - 第 2 關：有權限，含「站在這裡能不能用這支工具」
          - 第 3 關：參數格式
          - 第 4 關：行政前置條件
        """
        args = dict(args or {})
        tool = self.tools.get(tool_id)
        if tool is None:
            return self._log(tool_id, args,
                             fail("bad_arguments", ["no such tool"]))
        ctx = Ctx(self, tool, args)
        # 第 2 關的位置部分。擋下來的訊息固定，不提示該去哪裡。
        if not check_gate(ctx):
            return self._log(tool_id, args, fail("not_available_here"))
        # handler 執行前後各拍一次快照，把差異攤平成只追加的變更紀錄。
        # 放在這裡而不是每一支 handler 裡，所以 handler 不知道它存在。
        before = changelog.capture(self.state, self.records, self.world["slots"])
        try:
            # 順序：先驗參數格式，再驗前置條件。反過來的話，少傳一個
            # 必填欄位會被前置條件先攔下來，回傳 not_found 而不是
            # bad_arguments。
            validate_args(tool, args)     # 第 3 關：參數格式
            check_preconditions(ctx)      # 第 4 關：行政前置條件
            env = ok(self._dispatch(tool, ctx))
        except ToolError as e:
            env = fail(e.reason, e.fields)
        self._record_changes(tool_id, before)
        return self._log(tool_id, args, env)

    def _record_changes(self, tool_id: str, before: dict) -> None:
        """_record_changes（記下這次改了什麼）：比對前後兩份快照的差異。"""
        after = changelog.capture(self.state, self.records, self.world["slots"])
        for rec in changelog.diff(before, after):
            self.state["change_log"].append({
                "seq": self._seq + 1,          # 跟即將寫入的呼叫紀錄同號
                "at": self.today,
                "tool": tool_id,
                **rec,
            })

    def _log(self, tool_id: str, args: dict, env: dict) -> dict:
        """_log（寫一筆呼叫紀錄）。成功失敗都寫。
        fields 也要寫：它是 Agent 當下看到的那句話，不記的話事後只能
          從參數反推是哪一條規則擋的。
        """
        self._seq += 1
        self.state["tool_call_log"].append({
            "seq": self._seq,
            "tool": tool_id,
            "node": self.node_of(tool_id, args),
            "args": copy.deepcopy(args),
            "ok": env["ok"],
            "reason": env["reason"],
            "fields": list(env.get("fields") or []),
            "data": copy.deepcopy(env["data"]),
            "at": self.today,
        })
        return env

    def _dispatch(self, tool: dict, ctx: Ctx):
        """_dispatch（派工）：照工具檔寫的路徑找到實作函式並執行。
        工具檔裡寫的是一串用點號連起來的路徑，例如

            runtime.handlers.clinic.issue_hold_order

        拆成模組與函式名，載入模組、取出函式、呼叫它。新增一支工具
        不用改這裡，只要在 YAML 寫好路徑。
        """
        path = tool["handler"]
        mod_path, _, fn_name = path.rpartition(".")
        mod = self._handlers.get(mod_path)
        if mod is None:
            root = __package__.split(".")[0]
            mod = importlib.import_module(f"{root}.{mod_path}")
            self._handlers[mod_path] = mod
        fn = getattr(mod, fn_name, None)
        if fn is None:
            raise ToolError("bad_arguments", f"handler {path} 沒有實作")
        return fn(ctx)

    # ---------------- 便利方法 ----------------

    def advance(self, days: int = 1) -> None:
        """advance（把時鐘往前推幾天）。只有主迴圈在 end_turn 時呼叫。"""
        from .clock import shift
        self.today = shift(self.today, days)
        self.state["sim_clock"]["today"] = self.today
        # 每天開始時，位置重設回起始節點（Agent 自己的診所）。
        self.state["position"]["node"] = self.start_node

    def snapshot(self) -> dict:
        """snapshot（終態快照）：跑完之後交出來的那一包。
        刻意包含世界的 slots（已訂人數會變），不含指引（唯讀、不會變）。
        """
        return {"state": copy.deepcopy(self.state),
                "records": copy.deepcopy(self.records),
                "slots": copy.deepcopy(self.world["slots"]),
                "today": self.today}
