"""工具的實作
工具清單以 data/tools/<層級>/ 與各元素的 tools/ 為準

    world.py      世界裡的走動與查詢
                  move_to（移動到某個節點）
                  list_institutions（列出所有機構）
                  get_institution_info（查機構說明）
                  get_department_info（查科別說明）
                  get_service_info（查服務項目說明）
                  query_slots（查號源）
                  get_travel_time（查兩點之間的交通時間）
                  date_calc（日期加減與星期幾）

    clinic.py     轉出端診所能做的事
                  ask_patient（問病人問題）
                  consult_patient（把方案寄給病人並問他）
                  get_case_status（看目前的預約、醫囑、抽血單）
                  record_assessment（記下自己算出什麼、依據什麼）
                  issue_hold_order（開立停藥醫囑）
                  cancel_order（撤銷停藥醫囑或抽血單）
                  send_prep_instructions（寄準備說明單給病人）
                  update_patient_contact（更新病人電話）
                  escalate_to_physician（把臨床問題上報醫師）

    hospital.py   醫院端的預約作業
                  book_appointment（預約單一診次）
                  book_series（一次預約一整組相關診次）
                  reschedule_appointment（改期）
                  cancel_appointment（取消預約）
                  order_preop_lab（開立術前抽血單）

    referral.py   轉診單
                  create_referral（開立轉診單）
                  update_referral（更新轉診單）
                  withdraw_referral（撤回轉診單）

    records.py    get_patient_record（調閱病歷）
    retrieval.py  search_guidelines（檢索指引全文）
    core.py       end_turn（結束這一回合，時鐘往前走）

  - handler 只透過 Ctx 碰世界，拿不到地面真值（truth/）。用藥原因的
    分類對照表就在 truth/，所以沒有任何工具查得到它
  - handler 不做任何臨床檢查。要擋的條件全部在 executor.py 宣告，
    而且只有行政性的那幾條：號源滿了、日期不存在、不在正確地點
"""
