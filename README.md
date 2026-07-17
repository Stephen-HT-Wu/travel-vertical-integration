# 旅遊產業垂直整合 Agent Demo

一個展示 AI agent 系統如何垂直整合旅遊產業鏈的 demo：從人物設定出發，agent 鏈依序完成靈感探索 → 行程規劃 → 交通 → 住宿 → 餐飲 → 景點 → 活動 → 購物 → 導覽 → 評價與分享，並在行程確定後模擬一段「突發狀況 → 動態重新排程」的橋段。整趟流程由獨立的 `UserSimulatorAgent`（虛擬旅客）在每個確認點做決策，最後產出結構化的 `trip_log.json` 與一份單一檔案的 `dashboard.html`。

完整設計脈絡見 [`/Users/stephen/.claude/plans/agent-demo-modular-toucan.md`](file:///Users/stephen/.claude/plans/agent-demo-modular-toucan.md)。

## 這個 demo 誠實在說什麼

市面上很多「AI Travel Agent」其實只是 `LLM + 搜尋結果 + 聯盟連結`。這個 demo 刻意把每個階段對照一套七層旅遊 Agent 成熟度框架，並在 dashboard 上標示清楚：

- **靈感、餐飲、景點、購物** — 真實網路搜尋（可選限定可信網域），內容推薦類任務，成熟度高，本 demo 是真的在做。
- **交通、住宿、活動** — LLM 生成的模擬候選方案，清楚標示為 simulated。這是真正牽涉即時庫存/價格/交易的環節，成熟度低，本 demo **不**實際比價、訂位或付款——只做到「候選提案 + 人工確認」。
- **動態重新排程橋段** — 模擬一個行中突發狀況，展示 agent 重新排程的能力，同樣誠實標示為模擬情境。

整個程式碼中不存在任何訂位/付款工具，任何 agent 在任何 prompt 下都沒有能力執行真實交易。

## 安裝

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # 填入 ANTHROPIC_API_KEY
```

## 執行

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

檢查每個階段是否都有產出、`data_source` 是否符合預期（真實搜尋階段皆為 `real_search` 且有 `source_url`，模擬階段皆為 `simulated`）、動態重排橋段是否存在並被確認過、`hitl_log` 筆數是否合理、評分是否落在 `[1,5]`。

## Google Calendar 設定

互動版 demo 的「加入 Google Calendar」功能需要一組你自己在 Google Cloud Console 建立的 OAuth 2.0 用戶端憑證。這個專案不會、也不可能內建任何真實憑證——每一次寫入行事曆，都必須先經過 Google 自己的同意畫面，由你本人核准。

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
persona.py            # 人物設定 model + CLI 參數
schemas.py             # 所有結構化輸出 / trip log 的 Pydantic models
stage_metadata.py       # 每個階段對應七層成熟度框架的靜態標籤（供 dashboard 使用）
llm_client.py           # Anthropic SDK 薄封裝：結構化輸出呼叫、web_search 呼叫
agents/                 # 每個階段一個 agent 模組 + user_simulator_agent.py
orchestrator.py         # 串接所有階段、HITL 確認點、動態重排橋段
run_demo.py             # CLI 進入點
dashboard/              # Jinja2 樣板 + 渲染腳本
config/trusted_domains.json  # 可信網域清單範例（品牌無關，可自行替換）
tests/test_sanity.py    # trip_log.json 的結構性檢查
```

## 已驗證但需要 API key 才能實際執行

本機開發環境沒有 `ANTHROPIC_API_KEY`，因此以下項目已經過離線驗證：
- 所有模組語法正確、可正常 import（`python -m py_compile`、直接 import 全部模組）。
- 對照最新 Anthropic 文件確認 `client.messages.parse(output_format=...)` 與 `web_search_20250305` 工具的用法與已安裝 SDK 版本（`anthropic==0.117.0`）相符。
- 用一份手動建構、符合全部 schema 的合成 `trip_log.json`（不呼叫任何 API）驗證了：schema 驗證通過、`tests/test_sanity.py` 全部通過、`dashboard/render_dashboard.py` 產出的 HTML 可正確解析且所有區塊（人物設定、成熟度圖例、靈感、行程、六個候選階段、行中導覽、動態重排橋段、最終評價）都正確渲染。

尚未驗證：真正呼叫 Anthropic API 跑完整趟流程（含真實 web_search 搜尋結果、LLM 生成的候選方案品質）。設定好 `ANTHROPIC_API_KEY` 後執行「執行」一節的指令即可完成端到端驗證。
