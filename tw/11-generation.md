# 11. 生成新名字——溫度與隨機性

## 從模型到生成器

訓練完成後，我們有了一個「學會名字規律」的模型。

現在的問題：**如何讓它生成新名字？**

## 貪心解碼：每次選最高分的

最簡單的方法：每次選機率最高的 token：

```python
def greedy_decode(logits):
    return np.argmax(logits)  # 選最高的
```

優點：確定性輸出
缺點：缺乏多樣性，每次都一樣

## 溫度：控制隨機性

microgpt 使用**溫度抽樣**（Temperature Sampling）：

```python
temperature = 0.5  # 0 到 1 之間，越小越保守

# 對 logits 除以溫度再 softmax
probs = softmax([l / temperature for l in logits])
token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
```

### 溫度的直覺

| 溫度 | 效果 | 比喻 |
|-----|------|-----|
| 高（如 1.0）| 隨機性高，多樣性強 | 硬幣隨便翻 |
| 低（如 0.5）| 隨機性低，更保守 | 硬幣稍微不公平 |
| 接近 0 | 幾乎總是選最高的 | 硬幣永遠正面 |

### 數學解釋

原始 Softmax：

$$p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

加溫度後：

$$p_i = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}$$

- 當 $T = 1$：標準 Softmax
- 當 $T \to 0$：只有最大 logit 的機率趨近 1（貪心）
- 當 $T \to \infty$：所有 logit 趨近均勻分佈

## microgpt 的生成代碼

```python
temperature = 0.5

print("--- inference (new, hallucinated names) ---")
for sample_idx in range(20):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS  # 從 BOS 開始
    sample = []
    
    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)
        
        # 加溫度的 Softmax
        probs = softmax([l / temperature for l in logits])
        
        # 根據機率抽樣
        token_id = random.choices(range(vocab_size), 
                                   weights=[p.data for p in probs])[0]
        
        if token_id == BOS:  # 遇到 BOS 就停止
            break
        
        sample.append(uchars[token_id])
    
    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")
```

## 生成流程圖

```
BOS ──→ 模型 ──→ logits ──→ softmax(temperature) ──→ 抽樣 ──→ 'a'
                                                                 │
                                                                 ↓
BOS, a ──→ 模型 ──→ logits ──→ softmax(temperature) ──→ 抽樣 ──→ 'l'
                                                                           │
                                                                           ↓
BOS, a, l ──→ 模型 ──→ logits ──→ softmax(temperature) ──→ 抽樣 ──→ 'e'
                                                                                  │
                                                                                  ↓
BOS, a, l, e ──→ 模型 ──→ ... ──→ 抽樣 ──→ 'x'
                                                    │
                                                    ↓
BOS, a, l, e, x ──→ 模型 ──→ logits ──→ BOS ──→ 停止！
                                                    │
                                                    ↓
                                            輸出："alex"
```

## 生成結果

訓練 1000 步後，用溫度 0.5 生成的結果：

```
sample  1: ari
sample  2: karia
sample  3: alex
sample  4: michon
sample  5: ryan
sample  6: emily
sample  7: davi
sample  8: sophia
sample  9: andrew
sample 10: lexa
```

這些名字看起來像真實名字！模型學會了：
- 常見的字母組合（alex, emily, andrew）
- 名字的結構（多個樣本都像真實英文名）

## 不同溫度的效果

| 溫度 | 輸出特點 | 例子 |
|-----|---------|------|
| 0.2 | 非常保守，常見組合 | alex, emily, andrew |
| 0.5 | 平衡多樣性 | ari, karia, michon |
| 1.0 | 多樣性高，有些奇怪的組合 | axzh, kreoql, vnm |
| 2.0 | 幾乎隨機 | 可能完全不像名字 |

## 為什麼叫「 hallucinated」？

Karpathy 在代碼中用了「hallucinated」這個詞，意思是「幻想的」。

這些名字**不存在於訓練資料中**，但看起來合理——就像 AI 在「幻想」新名字。

同樣的道理，ChatGPT 有時會產生「幻覺」——看起來合理但實際上是錯誤的內容。

## 自迴歸生成：一步接一步

這種生成方式叫做**自迴歸**（Autoregressive）：

```
t=0: 輸入 BOS → 預測 w₁
t=1: 輸入 BOS, w₁ → 預測 w₂
t=2: 輸入 BOS, w₁, w₂ → 預測 w₃
...
```

每一步的輸出變成下一步的輸入。

優點：可以生成任意長度的序列
缺點：生成速度慢（無法平行）

## 與現代 LLM 的比較

| 特性 | microgpt | GPT-4 |
|-----|---------|-------|
| 生成方式 | 自迴歸 | 自迴歸 |
| 溫度控制 | 有 | 有 |
| 額外技巧 | 無 | Beam Search、Top-k、Top-p |

### Top-k 抽樣

```python
# 只從最高的前 k 個 token 中抽樣
top_k = 50
indices = np.argsort(logits)[-top_k:]  # 前 50 大的 index
probs_topk = softmax([logits[i] for i in indices])
token_id = random.choices(indices, weights=probs_topk)[0]
```

### Top-p (Nucleus) 抽樣

```python
# 從累積機率超過 p 的最小集合中抽樣
sorted_probs = sorted(probs, reverse=True)
cumsum = np.cumsum(sorted_probs)
mask = cumsum < 0.9  # 累積到 90%
token_id = random.choices(np.where(mask)[0])[0]
```

## 總結

這一章的核心概念：
- **貪心解碼**：每次選最高分的 token
- **溫度抽樣**：控制輸出的隨機性
- **自迴歸生成**：一步一步生成序列
- **幻覺**：生成看似合理但實際不存在/錯誤的內容

下一章，我們從 microgpt 升級到真正的 LLM——了解預訓練的力量！

---

*上一步：[10. 訓練循環](10-training_loop.md)*  
*下一步：[12. 預訓練的力量](12-pretraining.md)*
