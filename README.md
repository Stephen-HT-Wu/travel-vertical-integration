# 旅遊產業垂直整合 Agent Demo

一個展示 AI agent 系統如何垂直整合旅遊產業鏈的 demo：從人物設定出發，依序完成靈感探索 → 行程規劃 → 交通 → 住宿 → 餐飲 → 景點 → 活動 → 購物 → 導覽 → 評價與分享。有兩種執行方式：

- **CLI（全自動批次）**：`run_demo.py`，十個階段全部自動跑完，每個 HITL 確認點由獨立的 `UserSimulatorAgent`（虛擬旅客）依人物設定做決策，適合快速產生一份完整範例、或做離線可行性評估。
- **Web（對話式互動）**：`webapp.py`，靈感/行程/交通/住宿/活動這五個階段改成跟**真實使用者**多輪對話；交通/住宿/活動三個交易階段最後不是模擬下單，而是產生**真實 deep link**、開新分頁到真實 OTA 搜尋結果頁，使用者自己在對方網站完成——本 app 不追蹤、不碰金流。後續的餐飲/景點/購物/導覽/動態重排/評價，以及 Google Calendar 同步，可以透過本機設定整段關閉，只展示「靈感 → 導流」這段價值鏈。

完整設計脈絡見 [`/Users/stephen/.claude/plans/agent-demo-modular-toucan.md`](file:///Users/stephen/.claude/plans/agent-demo-modular-toucan.md)。

## 這個 demo 誠實在說什麼

市面上很多「AI Travel Agent」其實只是 `LLM + 搜尋結果 + 聯盟連結`。這個 demo 刻意把每個階段對照一套七層旅遊 Agent 成熟度框架，並在能力邊界上誠實：

- **靈感、餐飲、景點、購物** — 真實網路搜尋（可選限定可信網域），內容推薦類任務，成熟度高，本 demo 是真的在做。
- **交通、住宿、活動** — 候選方案本身仍是 LLM 生成、清楚標示為 `simulated`（不是真實的即時庫存或價格）；但對話版會把使用者導向**真實**的 Google 地圖／Booking.com／KKday 搜尋結果頁（deep link），讓使用者自己看到真實報價/庫存並自行完成——本 app 完全不碰金流，也不追蹤導流之後發生了什麼。
- **動態重新排程橋段**（CLI 全自動流程才有）— 模擬一個行中突發狀況，展示 agent 重新排程的能力，誠實標示為模擬情境。

整個程式碼中不存在任何訂位/付款工具，任何 agent 在任何 prompt 下都沒有能力執行真實交易；deep link 的網址一律由伺服器端用固定樣板組出（LLM 只提供搜尋關鍵字），組完後會驗證網域落在寫死的 vendor allowlist，LLM 產出的文字不可能讓連結跳到非預期的網站。

## 安裝

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # 填入 ANTHROPIC_API_KEY
```

## 執行（CLI，全自動批次）

```bash
python run_demo.py --age-group 26-35 --gender unspecified --location 台北 \
  --trip-length one-day --party-size 2 --site-mode unrestricted
```

執行完會產生：
- `output/trip_log.json` — 完整的結構化行程紀錄（含每個階段的輸出與所有 HITL 確認紀錄）
- `output/dashboard.html` — 單一檔案的視覺化 dashboard，可直接用瀏覽器開啟，也可以貼成 Claude Artifact

### 常用參數

| 參數 | 說明 |
|---|---|
| `--age-group` | `18-25` / `26-35` / `36-50` / `51+` |
| `--gender` | `male` / `female` / `unspecified` |
| `--location` | 出發地（必填），例如 `台北` |
| `--trip-length` | `half-day` / `one-day` / `multi-day` |
| `--days` | 天數，只有 `multi-day` 時有意義 |
| `--party-size` | 同行人數 |
| `--site-mode` | `unrestricted`（不限定網站）或 `allowlist`（限定可信網域） |
| `--site-list` | `--site-mode allowlist` 時使用的網域清單 JSON（預設 `config/trusted_domains.json`，品牌無關、可自行替換） |
| `--model` | 預設 `claude-opus-4-8` |
| `--output-dir` | 預設 `output` |

### 只重新產生 dashboard

```bash
python dashboard/render_dashboard.py output/trip_log.json output/dashboard.html
```

### 結構性檢查

```bash
python tests/test_sanity.py output/trip_log.json
```

檢查每個階段是否都有產出、`data_source` 是否符合預期、動態重排橋段是否存在並被確認過、`hitl_log` 筆數是否合理、評分是否落在 `[1,5]`。

## 執行（Web，對話式互動）

```bash
python webapp.py
```

打開 http://127.0.0.1:8000 ，設定人物後按「開始對話」。流程：靈感（多輪對話，可挑一個喜歡的）→ 行程（多輪對話，確認後鎖定）→ 交通/住宿/活動（各自對話挑一個候選，選定後立刻導流到真實網站的搜尋結果頁）→（若未關閉）自動接續餐飲/景點/購物/導覽/動態重排/評價 →（若未關閉）加入 Google Calendar。畫面上會即時顯示每個步驟花費的 tokens、金額（依 [`pricing.py`](pricing.py) 的公開牌價概算，非精確帳單）、時間，方便做可行性評估。

## 本機設定（`local_settings.json`）

有兩件事只能在本機覆蓋、不會進版控：**靈感搜尋預設限定的網域**、**要不要開啟後段自動流程與 Google Calendar 同步**。做法是在專案根目錄建立 `local_settings.json`（已加進 `.gitignore`），沒有這個檔案時行為與現況完全一樣（不限定搜尋、兩個功能都開）。

```bash
cp local_settings.example.json local_settings.json
```

```json
{
  "default_inspiration_domains": ["supertaste.tvbs.com.tw"],
  "enable_tail_pipeline": false,
  "enable_calendar_sync": false
}
```

- `default_inspiration_domains`：非空時，Web 版的搜尋模式會預設「限定可信網域清單」並帶入這裡的網域（畫面上會顯示一行小字說明這是本機預設）；留空 `[]` 就是現況（不限定，全網搜尋）。
- `enable_tail_pipeline` / `enable_calendar_sync`：設成 `false` 會讓對話版在交通/住宿/活動三個導流都完成後直接顯示「demo 完成」摘要，不會呼叫後段自動流程或顯示 Calendar 面板；伺服器端也會讓對應的 API（`/api/chat/{id}/tail-events`、`/api/calendar/*`）回傳 403，不只是前端隱藏。CLI（`run_demo.py`）不受這個設定影響，永遠跑完整十個階段。

## Google Calendar 設定

Web 版的「加入 Google Calendar」功能（`enable_calendar_sync=true` 時才會出現）需要一組你自己在 Google Cloud Console 建立的 OAuth 2.0 用戶端憑證。這個專案不會、也不可能內建任何真實憑證——每一次寫入行事曆，都必須先經過 Google 自己的同意畫面，由你本人核准。

1. 前往 [Google Cloud Console](https://console.cloud.google.com/)，建立一個新專案（或選擇既有專案）。

2. 啟用 Google Calendar API：左側選單 **APIs & Services → Library**，搜尋「Google Calendar API」，點進去按 **Enable**。

3. 設定 OAuth 同意畫面：**APIs & Services → OAuth consent screen**。
   - User Type 選 **External**（除非你有 Google Workspace 網域）。
   - 填必填欄位（App name、User support email、Developer contact）即可，不需要送審。
   - Scopes 這步可以先跳過（程式碼裡直接指定 `calendar.events` 這個 scope）。
   - Test users：把你自己要用來測試的 Google 帳號加進去。App 會停留在「Testing」狀態（未送審發布），只有加進 Test users 清單的帳號才能完成授權。

4. 建立憑證：**APIs & Services → Credentials → Create Credentials → OAuth client ID**。
   - Application type 選 **Web application**。
   - Name 隨意，例如「travel-agent-demo-local」。
   - **Authorized redirect URIs** 一定要加上：
     ```
     http://127.0.0.1:8000/api/calendar/oauth/callback
     ```
     這個網址要完全 match（含 http、port、路徑），不然 Google 會回傳 `redirect_uri_mismatch` 錯誤。

5. 建立完成後下載 JSON（Credentials 列表點進去 **Download JSON**），存成專案根目錄的：
   ```
   google_oauth_credentials.json
   ```
   這個檔名是程式碼裡寫死的路徑，已加進 `.gitignore`，不會被 commit。

6. 完成後，webapp 的「加入 Google Calendar」按鈕就會導向 Google 的同意畫面；登入並同意後會被導回本機的 `/api/calendar/oauth/callback`，用你剛核准的授權把確認過的行程逐一寫成事件到你的主要行事曆（primary calendar）。

**注意**：這裡用的 scope 是 `calendar.events`（只能建立/管理事件），不是完整的行事曆讀寫權限；App 在 Testing 狀態下 token 效期較短，但本機 demo 用途完全足夠；若要給別人展示，對方也需要被加進你 Google Cloud 專案的 Test users 清單才能完成授權——這是 Google 對 Testing 狀態 app 的限制。

## 專案結構

```
persona.py               # 人物設定 model + CLI 參數
schemas.py                # 所有結構化輸出 / trip log 的 Pydantic models
stage_metadata.py          # 每個階段對應七層成熟度框架的靜態標籤
pricing.py                 # 各模型的公開牌價，用於估算 token 花費
llm_client.py               # Anthropic SDK 薄封裝：結構化輸出呼叫、web_search 呼叫
deep_links.py                # 交通/住宿/活動導流用的真實 deep link 樣板（Google 地圖/Booking.com/KKday）
local_settings.py            # 本機設定載入（local_settings.json，gitignore 掉）
calendar_integration.py      # Google Calendar OAuth + 建立行事曆事件
agents/                      # 每個階段一個 agent 模組（.run() 給 CLI、.chat() 給 Web）+ user_simulator_agent.py
orchestrator.py               # CLI 全自動流程；也提供 run_tail_streaming() 給 Web 版的後段接續
chat_session.py                # Web 版的對話狀態機（靈感/行程/交通/住宿/活動 + 導流）
webapp.py                       # FastAPI app，Web 版進入點
webapp_static/index.html         # 對話式前端（純 vanilla JS，無框架）
run_demo.py                       # CLI 進入點
dashboard/                         # Jinja2 樣板 + 渲染腳本，CLI 與 Web 兩種 trip_log 都能渲染
config/trusted_domains.json        # 可信網域清單範例（品牌無關，可自行替換，checked-in）
local_settings.example.json        # 本機設定範本（checked-in，品牌無關）
tests/test_sanity.py               # trip_log.json 的結構性檢查
```

## 已驗證的項目

- CLI 全自動流程與 Web 對話式流程皆已用真實 `ANTHROPIC_API_KEY` 跑過完整端到端。
- 三個真實 deep link 的網址格式（Google 地圖大眾運輸、Booking.com 搜尋、KKday 商品搜尋）已在瀏覽器實際打開驗證，皆回傳有意義的真實結果；`deep_links.py` 的組裝邏輯已離線單元測試（host allowlist、空字串 fallback）。
- 真實 deep link 導流 + feature toggle 版本已用真實 API 跑過完整端到端：`local_settings.json` 設定 `supertaste.tvbs.com.tw` 後，靈感/餐飲類搜尋確實被限定在該網域；交通/住宿/活動三個階段的導流網址皆為正確 vendor 網域、query string 內容合理；`enable_tail_pipeline=false` 時，活動導流完成後 `next_phase` 正確直接變成 `"done"`，`dining`/`attractions`/`shopping`/`review` 等欄位保持空白；`/api/chat/{id}/tail-events` 與 `/api/calendar/*` 在關閉狀態下正確回傳 403。
- 過程中發現並修正一個真實問題：候選方案的結構化輸出偶爾會出現「技術上合法、但內容是佔位文字」的情況（`candidates` 空陣列、`agent_selected_candidate_id`/`deep_link_query` 直接寫 `"placeholder"`）——schema 限制本身不會擋下這種語意上的偷懶。已加強 prompt 明確禁止，並在 `agents/base_agent.py` 新增 `validate_candidate_turn()` 作為第二層防護，偵測到就丟出清楚的錯誤訊息，而不是靜默地讓使用者卡在沒有候選方案的畫面。
- Google Calendar OAuth 流程的程式碼已完成並可正常處理「憑證未設定」的錯誤情況；實際授權寫入行事曆的部分需要使用者自行建立 Google Cloud OAuth 憑證後才能驗證（見上方「Google Calendar 設定」）。
