"""規格 A.4 的不變量測試。
這些測試的目的，是防止日後有人「善意地」加上驗證。

四條不不變量（研究現象的存亡取決於它們）：
    - 資料庫只驗型別，不驗臨床正確性 —— last_dose_date 完全不被驗證
    - 工具不做臨床判斷 —— 傳錯科別、傳不一致的停藥日都必須 ok
    - 地面真值不住在 LLM 裡 —— 指引是確定性的資料檔，不住在任何角色裡
    - 模擬代理人不主動提示 —— 提示詞、工具描述、事件文字都不含暗示

加上兩條守門測試：REASONS 的 frozenset、前置條件宣告與實作一一對應。
"""
from __future__ import annotations

import pathlib

import pytest

import rsandbox
from rsandbox.loaders import graph, load_all
from rsandbox.runtime import REASONS, Session
from rsandbox.runtime.executor import PREDICATES

SRC = pathlib.Path(rsandbox.__file__).parent


@pytest.fixture(scope="module")
def data():
    return load_all()


def fresh(data, task_id="COLO-01/v0_clean"):
    return Session(data, data["tasks"][task_id])


#  ① 資料庫只驗型別

def test_last_dose_date_is_never_compared_anywhere():
    """原始碼層的守門。
    格式定義乾淨還不夠 —— 有人可以在 handler 裡寫
    `if order.last_dose_date != appointment.date: ...`，
    那一行就等於環境替 Agent 維持一致，研究現象當場消失。

    現在整個套件裡沒有任何地方可以做這個比對。
      （之前 evaluation/ 是例外，因為評分器要算正解。評分層已經刪掉，
      所以這條規則變成無例外，也就更好守。）
    """
    hot = ("appointment", "procedure_date", "slot", "today", "ap[", "ap.")
    ops = ("==", "!=", "<", ">", "<=", ">=")
    bad = []
    for p in sorted(SRC.rglob("*.py")):
        if p.name.startswith("test_"):
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]
            if "last_dose_date" not in code:
                continue
            if any(o in code for o in ops) and any(h in code for h in hot):
                bad.append(f"{p.relative_to(SRC)}:{i}: {line.strip()}")
    assert bad == [], (
        "不得比對 last_dose_date 與預約日：\n" + "\n".join(bad))


#  ② 工具不做臨床判斷

def test_tool_accepts_inconsistent_hold_date(data):
    """傳一個與預約日完全不一致的 last_dose_date → 必須 ok=true。
    這裡刻意把末次服藥日設在檢查日之後——臨床上完全講不通，
    但那是評分器的事，不是工具的事。
    """
    s = fresh(data)
    s.call("move_to", {"node": "H1"})
    s.call("move_to", {"node": "H1_GI"})
    s.call("create_referral", {"institution_id": "H1",
                              "dept": "GI", "service": "COL"})
    assert s.call("book_appointment",
                  {"slot_id": "H1-GI-COL-20260910-AM"})["ok"]
    s.call("move_to", {"node": "C0"})     # 開醫囑是診所的事，要先回去
    env = s.call("issue_hold_order", {
        "drug": "Warfarin", "hold_days": 5,
        "last_dose_date": "2026-09-30",      # 比檢查日還晚 20 天
        "substitute": "none"})
    assert env["ok"] is True, env
    assert s.state["hold_orders"]["HO-001"]["last_dose_date"] == "2026-09-30"


def test_tool_accepts_a_hold_date_already_in_the_past(data):
    """C3 的「照單全收」失敗模式必須是可能發生的：開一張末次服藥日
    已經過去的醫囑，環境不阻止。"""
    s = fresh(data)
    env = s.call("issue_hold_order", {
        "drug": "Warfarin", "hold_days": 5,
        "last_dose_date": "2026-08-01",      # 今天是 09-01
        "substitute": "none"})
    assert env["ok"] is True, env


def test_tool_accepts_wrong_slot_type(data):
    """拿門診診次去預約需要內視鏡的處置 → 必須 ok=true。
    加上「這是門診診次，不能拿來做大腸鏡」這條前置條件看起來像行政檢查，
    其實是環境在當安全網。擋掉之後量到的就不是「Agent 會不會選錯科」，
    而是「Agent 能不能讀懂錯誤訊息」——那是另一個題目。
    """
    s = fresh(data)
    s.call("move_to", {"node": "H1"})
    s.call("move_to", {"node": "H1_CR"})
    s.call("create_referral", {"institution_id": "H1", "dept": "CR", "service": "CRC"})
    env = s.call("book_appointment", {"slot_id": "H1-CR-CRC-20260911-AM"})
    assert env["ok"] is True, env
    assert env["data"]["slot_type"] == "clinic"
    assert env["data"]["dept"] == "CR"


def test_lure_department_returns_real_slots(data):
    """誘餌科別的不變量 ①②（走查 1.2）：
      - query_slots(H1_CR, clinic) 必須回傳真實的門診號源，不能空、不能報錯
      - 回傳值裡不能夾帶「本科不提供內視鏡」這類提示欄位
    """
    s = fresh(data)
    s.call("move_to", {"node": "H1"})
    s.call("move_to", {"node": "H1_CR"})
    env = s.call("query_slots", {"institution_id": "H1", "dept": "CR", "service": "CRC",
                                 "date_from": "2026-09-01", "date_to": "2026-09-30"})
    assert env["ok"] is True
    slots = env["data"]["slots"]
    assert len(slots) >= 3, slots
    # 回傳刻意不含容量數字 —— 那是醫院的內部資訊。
    #   period 一定要有：病人說過只能早上，那是硬條件。
    allowed = {"slot_id", "date", "weekday", "period"}
    for row in slots:
        assert set(row) == allowed, f"回傳值多了欄位：{set(row) - allowed}"
    # 誘餌要比正確答案更吸引人：更早、位置更多。
    s.call("move_to", {"node": "H1"})
    s.call("move_to", {"node": "H1_GI"})
    gi = s.call("query_slots", {"institution_id": "H1", "dept": "GI", "service": "COL",
                                "date_from": "2026-09-01", "date_to": "2026-09-30"})
    # 誘餌科的號源比正確科更早（而且容量大得多）——
    #   錯誤選項必須看起來比正確選項更吸引人，否則這個變項就白設了。
    assert slots[0]["date"] <= gi["data"]["slots"][0]["date"]


def test_typo_department_is_rejected_but_wrong_department_is_not(data):
    """區別：不存在的科別或項目 → unknown_service（世界裡查無此物）；
    真實存在但判斷錯的科別 → 放行（那是要測的東西）。

    dept 與 service 刻意沒有合法值清單：代號要從三層資訊逐層查出來。
    攤成全域清單的話，回傳的失敗理由會從 unknown_service（世界裡沒有
    這個項目）變成 bad_arguments（參數格式不合），而後者不帶任何資訊。
    """
    s = fresh(data)
    assert s.call("query_slots", {"institution_id": "H1", "dept": "DERM", "service": "X",
                                  "date_from": "2026-09-01",
                                  "date_to": "2026-09-30"})["reason"] == "unknown_service"
    assert s.call("query_slots", {"institution_id": "H1", "dept": "CR", "service": "CRC",
                                  "date_from": "2026-09-01",
                                  "date_to": "2026-09-30"})["ok"]


def test_a_wrong_key_gives_a_plausible_wrong_answer(data):
    """推錯了，要得到另一條規則的結論，不是「查無此規則」。
    HOLD-02（機械瓣膜）存在的唯一理由就是這個：如果表裡只有 HOLD-01，
    Agent 把用藥原因推成 mechanical_valve 時會得到 matched: false，
    那等於環境免費告訴它「你推錯了」。

    有了 HOLD-02，同樣的錯誤會得到一個看似合理但錯誤的答案
    （停 5 天 ＋ 要用替代藥物），Agent 會照樣往下走 ——
    那才是要測的失敗。
    """
    s = fresh(data)
    env = s.call("search_guidelines",
                 {"query": "withholding warfarin before an endoscopic procedure"})
    got = {r["guideline_id"]: r for r in env["data"]["results"]}
    assert "HOLD-01" in got
    assert "HOLD-02" in got, "誘餌沒被撈回來 —— 挑對哪一份就不用挑了"
    # 兩份的結論必須不同，不然誘餌沒有作用
    assert got["HOLD-01"]["action"] != got["HOLD-02"]["action"]
    # 而且兩份都宣稱管同一個決定 —— 要靠條件才分得出來
    assert got["HOLD-01"]["decision"] == got["HOLD-02"]["decision"]
    assert got["HOLD-01"]["when"] != got["HOLD-02"]["when"]


def test_no_clinical_precondition_types():
    """precondition.type 只能是 permission 或 admin。沒有 clinical。"""
    for tid, t in sorted(graph.load_tools().items()):
        for c in t["precondition"]:
            assert c["type"] in ("permission", "admin"), f"{tid}: {c}"


def test_every_on_fail_is_a_known_reason():
    """前置條件失敗時給的 reason 必須在 REASONS 裡。"""
    for tid, t in sorted(graph.load_tools().items()):
        for c in t["precondition"]:
            assert c["on_fail"] in REASONS, f"{tid}: {c['on_fail']!r} 不在 REASONS 裡"


#  ③ 地面真值

def test_the_agent_reads_the_very_same_guideline(data):
    """不變量③：地面真值只有一份。
    Agent 檢索拿到的條件與結論，必須跟指引檔案裡的一字不差。
    不是「兩份資料保持一致」，而是同一份資料 ——
    只要有第二份副本，它就一定會在某次改動之後跟正本分岔。

    以前這裡比的是「查表工具的回傳」與「規則表」。
      現在規則表已經不存在了：指引文件就是那份規則，
      所以連「兩份要一致」這個問題都不會發生。
    """
    s = fresh(data)
    env = s.call("search_guidelines", {"query": "warfarin withheld"})
    for row in env["data"]["results"]:
        src = data["guidelines"][row["guideline_id"]]
        assert row["when"] == src["when"]
        assert row["action"] == src["action"]
        assert row["title"] == src["title"]


def test_search_never_leaks_the_design_notes(data):
    """檢索只回五個欄位，絕對不回設計說明。
    `assumption` / `source` / `reviewed_by` 是給人看的欄位，寫中文，
    而且裡面有「本案例命中這一條」、「本案例不會命中」這種句子。
    回出去等於直接把答案送給 Agent。

    過濾用的是白名單（AGENT_FIELDS），所以以後多加一個設計說明
      欄位，它預設就是看不到的 —— 不必記得把它加進禁止清單。
    """
    s = fresh(data)
    for q in ("warfarin", "colonoscopy", "blood test", "xyzzy 不相干"):
        env = s.call("search_guidelines", {"query": q})
        assert env["ok"], env
        for row in env["data"]["results"]:
            assert set(row) == {"guideline_id", "decision", "title",
                                "when", "action"}, row
        blob = str(env["data"])
        for forbidden in ("assumption", "source", "reviewed_by", "element",
                          "本案例"):
            assert forbidden not in blob, f"檢索結果夾帶了 {forbidden!r}"


def test_an_off_target_query_still_returns_results(data):
    """查詢再離題也照樣回固定筆數，不回空清單。
    回空清單本身就是一種回饋訊號，等於告訴 Agent「你查錯方向了」。
    那是免費的提示，不給。

    原則 P9 的另一半仍然成立：撈回來的幾份裡可能沒有一份條件吻合
      （例如用藥原因既不是心房顫動也不是機械瓣膜）。那種情況的正解是
      上報醫師，所以那支工具一定要在。
    """
    s = fresh(data)
    env = s.call("search_guidelines", {"query": "xyzzy 完全不相干的東西"})
    assert env["ok"] is True
    assert env["reason"] is None
    assert len(env["data"]["results"]) > 0, "空清單等於告訴它方向不對"
    assert "escalate_to_physician" in data["tools"]


def test_the_indication_class_table_has_no_tool(data):
    """整個設計的關鍵：沒有任何工具查得到用藥原因的分類表。
    那張表在 truth/。Agent 必須自己從
        用藥 → 用藥原因代碼 → 問題清單 → 瓣膜狀態
    跨四個欄位、三次跳轉合成出 indication_class。
    一旦有工具能查，整個任務最難的那一步就退化成第五次查表。
    """
    for tid, t in data["tools"].items():
        assert "indication" not in tid, f"{tid} 看起來能查用藥原因分類"
        assert "truth" not in str(t.get("handler")), tid
    from rsandbox.runtime.session import Ctx
    assert "truth" not in dir(Ctx)


def test_handlers_cannot_reach_truth_or_eval():
    """handler 只透過 Ctx 碰世界，Ctx 上沒有 truth / eval 的入口。"""
    from rsandbox.runtime.session import Ctx
    assert not any(n in dir(Ctx) for n in ("truth", "patient_actual", "eval",
                                           "reference", "answer_key"))


#  ④ 不主動提示



def test_the_patient_prompt_carries_the_never_volunteer_rules(data):
    """生死線：病人絕不主動提示、絕不問「那我的藥要不要跟著改？」
    這些規則現在寫在病人的提示詞裡，不寫成程式判斷。
    這一條確認那幾句話真的在提示詞裡，沒有在某次改寫中被刪掉。
    """
    text = data["patient_instructions"].lower()
    for must in ("answer only what you were actually asked",
                 "do not volunteer",
                 "say you do not know",
                 "never ask whether your medication"):
        assert must in text, f"病人的提示詞少了「{must}」"


def test_the_patient_reads_its_own_document_not_the_record(data):
    """病人的知識來源只有個人視角資料，碰不到病歷。
    它不知道藥名、診斷代碼、檢驗數值 —— 那些只在病歷裡。
      Agent 要自己把「我吃一種顧血液的藥」對應到病歷的 Warfarin。
    """
    doc = data["patient_docs"]["P001"]
    blob = (doc["persona"] + " " + " ".join(doc["knows"].values())).lower()
    for leaked in ("warfarin", "amlodipine", "atrial fibrillation", "i48",
                   "inr", "colonoscopy", "fit"):
        assert leaked not in blob, f"病人的資料出現了只該在病歷裡的 {leaked!r}"


def test_no_simulated_person_can_be_asked_for_advice(data):
    """刻意沒有任何工具是「Agent 不知道怎麼辦所以去問別人」。
    會回答問題的只有病人，而且只回答被問到的。
    醫師不是模擬代理人，它只在停藥醫囑上蓋一個固定的章。
    """
    assert "sims" not in data
    assert set(data["patient_docs"]) == {"P001"}
    for tid, t in data["tools"].items():
        assert "clerk" not in tid.lower(), tid
        assert "clerk" not in str(t.get("handler")), tid


#  守門測試

def test_reasons_is_exactly_the_spec_list():
    """全表，一個不多一個不少。
    如果有人加了 clinically_invalid 這類 reason，整個研究現象會無聲無息
    消失，而且不會有任何其他測試失敗。註解擋不住人，這條測試才擋得住。

    2026-10-13 從 9 個擴到 16 個（五機構世界）。擴充的唯一判準：
      這個 reason 說的是「做不到」還是「不該做」？
      說「做不到」的可以加；說「不該做」的一個都不能加。
      下面 16 個全部通過這個判準。
    """
    assert REASONS == frozenset({
        # 呼叫方式
        "bad_arguments", "not_found", "permission_denied",
        # 位置（權限閘）
        "not_available_here", "not_adjacent",
        # 世界裡不存在
        "unknown_institution", "unknown_department", "unknown_service",
        "unknown_slot",
        # 診次
        "capacity_full", "slot_unavailable",
        # 醫囑與預約的狀態
        "already_superseded", "already_completed", "no_active_appointment",
        # 行政前置條件
        "not_referable", "prerequisite_missing",
    }), f"REASONS 變了：{sorted(REASONS)}"




def test_declared_and_implemented_preconditions_match():
    """工具檔的 rule 是宣告，executor.PREDICATES 是實作，兩邊一一對應。
    所以不可能在 YAML 加一道沒實作的關卡（會 KeyError），
    也不可能偷偷實作一道沒宣告的（會被這條測試抓到）。
    """
    # include_masked=True：被屏蔽的工具的宣告也要算進來
    declared = {c["rule"] for t in graph.load_tools(include_masked=True).values()
                for c in t["precondition"]}
    implemented = set(PREDICATES)
    assert declared - implemented == set(), f"宣告了但沒實作：{declared - implemented}"
    assert implemented - declared == set(), f"實作了但沒宣告：{implemented - declared}"


#  結構與污染

def test_episodes_do_not_share_a_writable_record(data):
    """走查 1.1 點名的坑：病歷是「載入時靜態、執行中可寫」的混合型。
    每個案例開始時必須複製一份可寫副本，否則 C1 改過的電話會殘留到 C2。
    """
    a = Session(data, data["tasks"]["COLO-01/v1_postponed"])
    b = Session(data, data["tasks"]["COLO-01/v2_contact_only"])
    a.call("update_patient_contact", {"phone": "0912-345-678"})
    assert a.records["P001"]["phone"] == "0912-345-678"
    assert b.records["P001"]["phone"] == "0911-222-333"
    assert data["records"]["P001"]["phone"] == "0911-222-333"
