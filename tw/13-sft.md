# 13. SFT：監督式微調

## 從 Base 模型到 Instruction 模型

預訓練的 Base 模型很強，但不太會聽懂人類的指令：

```
Base 模型輸入："解釋量子糾纏"
Base 模型輸出："量子糾纏是量子力學中..."（可能繼續寫"量子糾纏是..."
（它只是在「補完」文字，不是在「回答」問題）
```

**SFT（Supervised Fine-Tuning，監督式微調）** 就是把 Base 模型變成能聽懂指令的模型。

## 什麼是 SFT？

SFT 的核心思想很簡單：

**收集人類寫的高品質問答對，用這些資料微調 Base 模型。**

```
問題：什麼是量子糾纏？
答案：量子糾纏是兩個或多個粒子之間的一種特殊關聯...
```

讓模型學習：
- 給定問題 → 生成答案
- 遵循指令 → 輸出有用的回覆

## SFT 的資料格式

SFT 使用的資料叫做 **Instruction-Tuning 資料集**：

```json
{
  "instruction": "解釋什麼是量子糾纏",
  "input": "",
  "output": "量子糾纏是量子力學中的一種現象..."
}
```

或者更複雜的格式：

```json
{
  "instruction": "把以下文字翻譯成英文",
  "input": "你好，世界！",
  "output": "Hello, World!"
}
```

## 指令資料集的規模

| 資料集 | 規模 | 用途 |
|-------|------|-----|
| FLAN | 1M+ | 各種任務 |
| Alpaca | 52K | 通用指令 |
| Vicuna | 70K | 對話 |
| WizardLM | 700K | 複雜指令 |

## SFT 的訓練過程

```python
# 1. 載入預訓練好的 Base 模型
model = load_base_model("gpt-3")

# 2. 準備指令資料
dataset = load_instruction_dataset("alpaca_data.json")

# 3. 用指令資料微調
for epoch in range(3):
    for item in dataset:
        # 前向傳遞：計算模型預測
        # 目標：讓預測接近人工寫的答案
        loss = compute_loss(model, item)
        
        # 反向傳播 + 更新參數
        loss.backward()
        optimizer.step()
```

## 訓練細節

### 學習率
比預訓練低很多，通常是預訓練學習率的 1/10 到 1/5。

```python
# 預訓練學習率：0.0001
# SFT 學習率：0.00001 ~ 0.00002
```

### Epoch 數
通常很少（1-5 個 epoch），避免過擬合。

### 資料品質
品質比數量更重要。高品質的 1 萬條比低品質的 100 萬條更有效。

## SFT 的效果

| 能力 | Base 模型 | SFT 後 |
|-----|----------|--------|
| 聽懂指令 | 差 | 好 |
| 回答質量 | 不穩定 | 穩定 |
| 格式正確性 | 差 | 好 |
| 安全性 | 差 | 較好 |

## 為什麼 SFT 不夠？

SFT 很有效，但有一些限制：

1. **需要大量人工標註**：成本高、耗時長
2. **難以擴展**：新任務需要新資料
3. **風格問題**：模型可能只是「模仿」訓練資料的回答風格
4. **安全性有限**：SFT 不能完全解決有害輸出

## Self-Instruct：減少人工標註

為了減少人工標註成本，研究者提出了 **Self-Instruct** 方法：

1. 用少量人類寫的範例
2. 讓模型生成更多指令資料
3. 用這些資料微調

```python
# 用少量範例提示模型生成更多
seed_examples = [
    {"instruction": "什麼是...?", "output": "..."},
    {"instruction": "如何...?", "output": "..."},
]

# 讓模型生成新指令
new_instructions = model.generate(
    prompt=f"根據以下範例，生成更多類似的問答：\n{seed_examples}"
)
```

Alpaca 就是用 Self-Instruct 方法從 GPT-3.5 生成資料的。

## 總結

這一章的核心概念：
- **SFT 目的**：把 Base 模型變成聽懂指令的模型
- **訓練資料**：人類寫的高品質問答對
- **訓練方式**：監督式學習，預測答案
- **Self-Instruct**：用模型生成訓練資料

下一章，我們看 RLHF——如何讓模型從人類回饋中學習，變得更安全、更有幫助！

---

*上一步：[12. 預訓練的力量](12-pretraining.md)*  
*下一步：[14. RLHF：從人類回饋中學習](14-rlhf.md)*
