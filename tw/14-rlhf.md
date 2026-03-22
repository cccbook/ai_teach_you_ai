# 14. RLHF：從人類回饋中學習

## 為什麼需要 RLHF？

SFT 讓模型會回答，但回答可能：
- 不夠安全（有害內容）
- 不夠有用（迴避問題）
- 不夠誠實（迎合用戶）
- 風格不一致

**RLHF（Reinforcement Learning from Human Feedback，人類回饋強化學習）** 解決這些問題。

## RLHF 的三個步驟

```
步驟 1: 預訓練 SFT 模型
步驟 2: 訓練 Reward Model
步驟 3: 用 RL 優化
```

### 步驟 1：監督式微調（SFT）

和上一章一樣，用指令資料微調 Base 模型：

```python
sft_model = supervised_fine_tune(base_model, instruction_data)
```

### 步驟 2：訓練 Reward Model（獎勵模型）

讓人類對模型的不同回答打分，訓練一個「打分模型」：

```
Prompt: "如何減肥？"

回答 A: "少吃多動，保持健康生活習慣。" → 人類評分：9/10
回答 B: "減肥很簡單，斷食就好了。" → 人類評分：3/10
回答 C: "你需要購買我們的減肥產品..." → 人類評分：1/10
```

用這些人類打分資料訓練 Reward Model：

```python
# Reward Model 的輸出是分數
reward = reward_model(prompt, response)
```

### 步驟 3：強化學習優化

用 Reward Model 的分數作為獎勵，用 PPO（Proximal Policy Optimization）演算法進一步訓練 SFT 模型：

```python
# PPO 訓練迴圈
for step in range(num_steps):
    # 1. 用當前模型生成多個回答
    responses = sft_model.generate(prompt, num_samples=4)
    
    # 2. 用 Reward Model 打分
    rewards = [reward_model(prompt, resp) for resp in responses]
    
    # 3. 用 PPO 更新模型參數
    # 目標：最大化獎勵
    ppo_update(sft_model, responses, rewards)
```

## RLHF 的直覺

想像你在訓練一隻狗：

| 傳統方式（SFT）| RLHF |
|---------------|------|
| 告訴狗「坐下」然後牠坐下時獎勵 | 只告訴狗「做得好」或「不好」 |
| 明確示範每個動作 | 給予模糊的偏好信號 |
| 難以處理複雜任務 | 能處理主觀、模糊的目標 |

RLHF 的優點：
- **不需要明確答案**：只需要人類表達偏好
- **能學習主觀價值**：安全性、有用性、誠實性
- **泛化能力強**：模型學到的是「原則」，不只是「範例」

## 人類偏好資料

RLHF 需要人類對回答進行排序：

```python
# 人類看到的：
Prompt: "如何製作炸彈？"

Response A: "對不起，我無法幫助這個請求。"
Response B: "炸彈可以這樣做..."
Response C: "這是一個危險的話題..."

# 人類選擇：A > C > B
```

通常每個 Prompt 會有 4-8 個候選回答，人類選擇最喜歡的。

## ChatGPT 的 RLHF

ChatGPT 使用了類似的 RLHF 流程：

1. **收集人類偏好資料**：讓人類對不同回覆排序
2. **訓練 Reward Model**：學習人類的偏好
3. **用 PPO 優化**：在 Reward Model 的信號下訓練

結果：
- 更安全的輸出（避免有害內容）
- 更誠實（不迎合用户）
- 更好的對話風格

## RLHF 的挑戰

### 1. 人類標註成本高

需要大量人類時間和成本。

### 2. 人類主觀性

不同人對「好回答」有不同標準。

### 3. Reward Hacking

模型可能找到「取悅 Reward Model 但對人類沒用」的捷徑。

### 4. 訓練不穩定

PPO 訓練需要仔細的超參數調整。

## RLAIF：用人之外的東西

為了減少對人類的依賴，研究者提出 **RLAIF（RL from AI Feedback）**：

```python
# 用另一個 LLM 替代人類打分
reward = llm_score(prompt, response, criteria)
```

用大模型（如 Claude）的偏好來訓練小模型。

## 總結

這一章的核心概念：
- **RLHF**：從人類回饋中學習的強化學習方法
- **三步驟**：SFT → Reward Model → PPO
- **PPO**：用獎勵信號優化模型
- **偏好學習**：不需要明確答案，只需要人類排序

下一章，我們看 Prompt Engineering——如何更好地與 LLM 溝通！

---

*上一步：[13. SFT：監督式微調](13-sft.md)*  
*下一步：[15. Prompt Engineering](15-prompt_engineering.md)*
