# 18. 上下文窗口與 Agent

## 上下文窗口：AI 的「記憶力」

**上下文窗口（Context Window）** 是 LLM 能夠處理的最大文字長度。

### 為什麼有限制？

Transformer 的注意力機制需要計算每個 token 與所有其他 token 的關係：

$$O(n^2 \cdot d)$$

其中 $n$ 是序列長度，$d$ 是維度。隨著 $n$ 增長，計算量以平方成長。

| 模型 | 上下文長度 | 約等於 |
|-----|-----------|-------|
| GPT-3.5 | 4K - 16K tokens | ~3000 - 12000 中文字 |
| GPT-4 | 8K - 128K tokens | ~6000 - 96000 中文字 |
| Claude 2 | 200K tokens | ~150000 中文字 |

### 上下文窗口的影響

```python
# ✅ 在上下文窗口內
prompt = """
以下是客戶投訴的完整記錄（5000 字）：
[5000 字內容]
請總結這個投訴的重點。
"""

# ❌ 超出上下文窗口
prompt = """
以下是公司十年的所有客戶來往郵件（100 萬字）：
[100 萬字內容]
請分析客戶滿意度的趨勢。
"""
```

## 如何應對有限的上下文？

### 1. 摘要壓縮

```python
# 原始：100 頁文件
# 步驟 1：讓 AI 總結每個章節
# 步驟 2：基於摘要回答問題
```

### 2. RAG（檢索增強生成）

```
用戶問題 → 檢索相關文件 → 加入上下文 → AI 回答
```

```python
# 1. 把長文檔分割成小塊
chunks = split_document("長文檔.txt", chunk_size=500)

# 2. 建立向量索引
index = build_vector_index(chunks)

# 3. 檢索相關內容
relevant_chunks = search(index, "用戶問題")

# 4. 組合成 prompt
prompt = f"""
相關文件內容：
{relevant_chunks}

問題：{user_question}
"""
```

### 3. 分段處理

```python
# 把長任務拆成多個短任務
task = "分析這本書的所有角色"
chapters = ["第1章", "第2章", ...]

for chapter in chapters:
    analysis = ai.analyze(chapter, "分析這個章節的角色")
    all_analyses.append(analysis)

final_summary = ai.summarize(all_analyses, "總結所有角色分析")
```

## Agent：讓 AI 能夠行動

**AI Agent** 是一種能夠自主規劃、使用工具、執行行動的 AI 系統。

### Agent 的核心能力

```
┌─────────────────────────────────────┐
│           AI Agent                   │
├─────────────────────────────────────┤
│  感知 (Perception)                   │
│    理解用戶輸入、環境資訊            │
├─────────────────────────────────────┤
│  規劃 (Planning)                     │
│    分解任務、制訂行動計劃            │
├─────────────────────────────────────┤
│  行動 (Action)                        │
│    使用工具、呼叫 API、執行操作       │
├─────────────────────────────────────┤
│  記憶 (Memory)                        │
│    儲存對話歷史、學習經驗            │
└─────────────────────────────────────┘
```

### Tool Use：讓 AI 使用工具

LLM 本身只能生成文字，但通過 **Function Calling** 可以使用外部工具：

```python
# 定義工具
tools = [
    {
        "name": "search",
        "description": "搜尋網路獲取最新資訊",
        "parameters": {"query": "string"}
    },
    {
        "name": "calculate",
        "description": "執行數學計算",
        "parameters": {"expression": "string"}
    },
    {
        "name": "send_email",
        "description": "發送電子郵件",
        "parameters": {"to": "string", "content": "string"}
    }
]

# AI 可以調用這些工具
user_input = "搜尋最新的人工智慧新聞，然後用郵件發送摘要給老師"

# AI 的 plan：
# 1. 調用 search({"query": "最新人工智慧新聞"})
# 2. 整理摘要
# 3. 調用 send_email({"to": "老師", "content": "摘要"})
```

### ReAct：思考 + 行動 + 觀察

讓 AI 在思考過程中結合行動：

```python
prompt = """
你是一個旅遊助手。對於每個問題：
1. Thought：思考要做什麼
2. Action：執行一個行動（使用工具）
3. Observation：觀察結果
4. 根據觀察繼續思考或給出最終答案

用戶問題：我本週末要去台北，想知道天氣如何，並推薦一個附近景點。

Thought: 我需要先查詢台北的天氣預報
Action: call_weather(city="台北")
Observation: 天氣預報顯示週末晴朗，氣溫 25-30 度
Thought: 天氣很好，我可以推薦戶外景點
Action: call_search(query="台北週末戶外景點推薦")
Observation: 找到以下景點：陽明山國家公園、淡水老街、士林夜市...
Final: 根據天氣和搜尋結果，推薦您週末去陽明山國家公園...
"""
```

### 自動駕駛汽車的 AI Agent

```python
# 感知
perception = {
    "cameras": "前方有車",  # 影像辨識
    "lidar": "距離前車 30 公尺",  # 測距
    "maps": "在忠孝東路上"  # 定位
}

# 規劃
plan = {
    "goal": "保持在車道中央，前進 100 公尺後右轉",
    "speed": "維持時速 40 公里",
    "emergency_plan": "如果有人闖入，立即剎車"
}

# 行動
actions = {
    "steering": "保持直行角度",
    "throttle": "維持當前油門",
    "brake": "待命"
}
```

## Agent 的應用場景

| 應用 | 描述 |
|-----|------|
| 個人助理 | 幫助處理郵件、日程、任務 |
| 程式設計助手 | 自動寫 code、debug、重構 |
| 研究助手 | 搜尋文獻、整理資料、撰寫報告 |
| 自動化流程 | 自動填表、處理文件、審批流程 |

## Agent 的挑戰

### 1. 規劃失敗

複雜任務的規劃可能出錯：

```python
# 問題：組裝家具
# 實際上：零件順序錯誤導致無法組裝
```

### 2. 工具使用錯誤

LLM 可能調用錯誤的工具或參數：

```python
# 意圖：搜尋天氣
# 實際：調用了錯誤的 API，返回錯誤結果
```

### 3. 錯誤累積

每一步的小錯誤可能累積成大問題：

```
步驟 1：小錯誤 → 
步驟 2：基於錯誤的輸入 → 
步驟 3：更大的錯誤 → 
... → 
最終：完全錯誤的結果
```

### 4. 安全風險

能執行行動的 Agent 可能造成更大的危害：

```python
# 問題：AI 自主決定發送大量郵件
# 結果：被當成垃圾郵件發送者、造成損失
```

## 總結

這一章的核心概念：
- **上下文窗口**：LLM 的「記憶力」限制
- **應對方法**：摘要、RAG、分段處理
- **Agent**：能感知、規劃、行動的 AI 系統
- **Tool Use**：讓 AI 使用外部工具
- **挑戰**：規劃失敗、錯誤累積、安全風險

下一章，我們展望 AI 的未來！

---

*上一步：[17. 幻覺、偏見與安全](17-hallucination_safety.md)*  
*下一步：[19. AI 的未來](19-future.md)*
