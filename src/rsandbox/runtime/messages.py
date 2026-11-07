"""環境會對 Agent 說的每一句話，全部集中在這裡。
handler 與主迴圈都從這張表取字串，不在各自的程式裡臨時組句子，
所以這個檔就是 Agent 會聽到的全部句子。

每一回合的觀察只講三件事：今天幾號、發生了什麼事、那件事的原文。
"""
from __future__ import annotations

# TEMPLATES（文字模板表）。{} 裡的東西會在送出前被換成實際的值。
TEMPLATES = {
    #  工具造成的副作用，病人會收到的通知
    # 改期之後通知病人新日期。只有日期，一個字都不提用藥。
    "appointment_moved": "Your appointment has been moved to {date}.",

    # 病人沒聽懂問題時的回答（問句沒有觸發任何一條他知道的事實）。
    #   - 措辭是我沒聽懂你在問什麼，不是這件事我不確定，這樣 Agent
    #     才有機會換個問法重問
    #   - 一樣不提示該問什麼
    "patient_unsure": "Sorry, I'm not sure what you're asking.",
                                                    # 抱歉，我不太確定你在問什麼

    # 這一科存在，但沒有公布任何資訊。
    # 跟這一科不存在是兩件事：後者在前置條件就被擋掉了。
    "dept_empty": "No information published for this department.",

    # 上報醫師之後收到的回覆。所有任務都是同一句。
    # 它明說醫師不會回覆，否則 Agent 會一直等下去。
    "escalated": ("Noted. The physician reviews escalated questions separately "
                  "and will not reply during this case. Continue with any work "
                  "that does not depend on this question."),

    # 轉診被接受。只說接受了，不說接下來該做什麼。
    "referral_accepted": "Referral {referral_id} accepted by {institution}.",

    # 停藥醫囑上的核可。無條件通過，所以這一句是常數：不評論內容、
    # 不提日期、不提規則。即使 Agent 開的是一張末次服藥日在三天前的
    # 醫囑，回的還是這一句。
    "doctor_approved": "Approved.",

    #  主迴圈每一回合送給 Agent 看的觀察
    # 回合的開頭。系統提示詞已經說過每次觀察都會告訴你今天幾號。
    "obs_today": "Today is {date}.",

    # 第一回合的案件簡報。
    #   - 只說有一個病人的轉診案開啟了，不說要做什麼
    #   - 只給病人編號，不附病歷全文（見 loop.py）
    "obs_opened": "A referral case has been opened for patient {patient_id}.",

    # 病人來電。{content} 是任務檔裡寫的原文，逐字轉述，
    # 不摘要、不加註、不解讀。
    "obs_patient_call": "The patient called and said: {content}",

    # 系統通知（例如某個時段被取消）。
    "obs_system_notice": "System notice: {content}",

    # 系統提醒（例如例行的確認通知）。
    "obs_system_reminder": "Reminder: {content}",

    # 什麼事都沒發生的那一天。刻意只有日期，不問還有什麼要做嗎。
    "obs_quiet": "",
}

# EVENT_TEMPLATE：任務檔裡的事件型別 -> 要用上面哪一個模板
EVENT_TEMPLATE = {
    "patient_call": "obs_patient_call",
    "system_notice": "obs_system_notice",
    "system_reminder": "obs_system_reminder",
}


def render(key: str, **kw) -> str:
    """render（取出一個模板、把空格填上），回傳最後給 Agent 看的那句話。"""
    return TEMPLATES[key].format(**kw)
