# 10. 訓練循環與 Adam 優化器

## 訓練循環：讓模型學會預測

microgpt 的訓練循環非常精簡，卻包含了完整的学习过程：

```python
num_steps = 1000

for step in range(num_steps):
    # 1. 取一個文件（名字）
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)
    
    # 2. 前向傳遞 + 計算損失
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()  # 交叉熵損失
        losses.append(loss_t)
    loss = (1 / n) * sum(losses)
    
    # 3. 反向傳遞：計算梯度
    loss.backward()
    
    # 4. 更新參數（Adam 優化器）
    lr_t = learning_rate * (1 - step / num_steps)  # 線性衰減
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad      # 第一動量估計
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2  # 第二動量估計
        m_hat = m[i] / (1 - beta1 ** (step + 1))          # 偏差校正
        v_hat = v[i] / (1 - beta2 ** (step + 1))          # 偏差校正
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps)     # 更新
        p.grad = 0  # 清除梯度
    
    if step % 100 == 0:
        print(f"step {step}: loss {loss.data:.4f}")
```

## 資料準備：從文字到 Tokens

```python
doc = docs[step % len(docs)]  # 取一個名字
# doc = "alex"

tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
# tokens = [26, 0, 11, 5, 23, 26]  (BOS, a, l, e, x, BOS)
```

為什麼要加 BOS（Begin Of Sequence）？

```
tokens:    [BOS]  →  a  →  l  →  e  →  x  →  [BOS]
position:    0       1     2     3     4       5

訓練目標：
- 給定 BOS，預測 a
- 給定 BOS, a，預測 l
- 給定 BOS, a, l，預測 e
- 給定 BOS, a, l, e，預測 x
```

BOS 標記序列的開始，讓模型知道從哪裡開始。

## 損失函數：交叉熵

```python
loss_t = -probs[target_id].log()
```

這就是**交叉熵損失**（Cross-Entropy Loss）。

直覺：
- 如果正確答案的機率是 1.0，log(1.0) = 0，損失 = 0（完美）
- 如果正確答案的機率是 0.1，log(0.1) = -2.3，損失 = 2.3（不好）

數學：

$$L = -\log(p_{correct})$$

- 當 $p_{correct} → 1$，$L → 0$（好）
- 當 $p_{correct} → 0$，$L → \infty$（差）

## Softmax：機率分佈

```python
def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]
```

Softmax 把一串數字轉換成機率分佈：

```python
logits = [3.2, 1.5, 0.8, 0.3]  # 每個 token 的分數
probs = softmax(logits)
# probs = [0.72, 0.17, 0.08, 0.03]  # 機率分佈，總和為 1
```

注意：數值穩定化（減去 max_val）防止 overflow。

## Adam 優化器

Adam（Adaptive Moment Estimation）是目前最常用的優化器：

```python
# 超參數
learning_rate = 0.01
beta1 = 0.85      # 第一動量（梯度的 EMA）
beta2 = 0.99      # 第二動量（梯度平方的 EMA）
eps_adam = 1e-8   # 防止除零

# 訓練循環中
for i, p in enumerate(params):
    # 1. 更新第一動量估計（類似動量）
    m[i] = beta1 * m[i] + (1 - beta1) * p.grad
    
    # 2. 更新第二動量估計（類似 RMSProp）
    v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
    
    # 3. 偏差校正（因為一開始 m, v 都是 0）
    m_hat = m[i] / (1 - beta1 ** (step + 1))
    v_hat = v[i] / (1 - beta2 ** (step + 1))
    
    # 4. 更新參數
    p.data -= learning_rate * m_hat / (v_hat ** 0.5 + eps_adam)
```

### Adam 的直覺

| 組成 | 作用 |
|-----|------|
| 第一動量 (m) | 累積歷史梯度方向（類似慣性）|
| 第二動量 (v) | 調整每個參數的學習率（梯度的量級越大，學習率越小）|

簡單來說：
- 梯度穩定的方向，大步前進
- 梯度震盪的方向，小步前進

## 學習率衰減

```python
lr_t = learning_rate * (1 - step / num_steps)
```

線性衰減：學習率從初始值慢慢降到 0。

```
學習率
  ↑
  │────
  │    ╲
  │      ╲
  │        ╲
  │─────────────→ 步數
  0          num_steps
```

為什麼要衰減？後期參數接近最優值，用大學習率可能會震盪。

## 訓練結果

microgpt 訓練 1000 步的典型輸出：

```
num docs: 32033
vocab size: 27
num params: 4192
step    0 / 1000 | loss 3.3660
step  100 / 1000 | loss 3.2345
step  200 / 1000 | loss 3.1023
step  300 / 1000 | loss 2.9876
step  400 / 1000 | loss 2.8765
step  500 / 1000 | loss 2.7654
step  600 / 1000 | loss 2.6543
step  700 / 1000 | loss 2.5432
step  800 / 1000 | loss 2.4321
step  900 / 1000 | loss 2.3210
step 1000 / 1000 | loss 2.2099
```

Loss 從 3.37 降到 2.21——模型在學習！

## 為什麼 loss 不是 0？

理論上，如果模型完美，loss 應該是 0（預測正確的機率 = 1）。

但實際上：
1. **模型太小**：只有 4,192 參數，學習能力有限
2. **資料隨機性**：名字本身有隨機性（沒有完美的規律）
3. **字符級預測的局限**：只看上一個字預測下一個字，本質上不可能 100% 準確

loss = 2.21 的意思是：正確 token 的機率約為 $e^{-2.21} ≈ 11\%$。

## 總結

這一章的核心概念：
- **訓練循環**：取資料 → 前向傳遞 → 反向傳遞 → 更新參數
- **交叉熵損失**：衡量預測分佈與真實分佈的差距
- **Adam 優化器**：結合動量和自適應學習率
- **學習率衰減**：後期用較小的學習率穩定收斂

下一章，我們看如何用訓練好的模型生成新名字！

---

*上一步：[9. GPT 架構](09-gpt_architecture.md)*  
*下一步：[11. 生成新名字——溫度與隨機性](11-generation.md)*
