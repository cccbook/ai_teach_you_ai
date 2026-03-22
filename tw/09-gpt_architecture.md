# 9. GPT 架構：Embedding → Transformer 層 → LM Head

## GPT 的任務：下一個 Token 預測

GPT（Generative Pre-trained Transformer）的任務非常簡單：

**給定前面所有的 token，預測下一個 token。**

```
輸入：I like to
輸出：eat

輸入：The sky is
輸出：blue
```

訓練時，讓模型學習這個映射。

## 三個主要階段

```
Token IDs → Token Embedding → + Position Embedding → Transformer Layers → LM Head → Logits
                    ↑                                              ↓
              查表取得                                    預測下一個 Token
```

## 1. Token Embedding

把 token ID 轉換成向量：

```python
# state_dict['wte']: vocab_size × n_embd 的矩陣
tok_emb = state_dict['wte'][token_id]  # 取得第 token_id 行的向量
# tok_emb: [v0, v1, v2, ..., v15]（16 維向量）
```

直覺：每個字元有自己的「性格向量」。

## 2. Position Embedding

讓模型知道 token 的位置：

```python
# state_dict['wpe']: block_size × n_embd 的矩陣
pos_emb = state_dict['wpe'][pos_id]  # 取得第 pos_id 行的向量
```

## 3. 融合：Token + Position

```python
x = [t + p for t, p in zip(tok_emb, pos_emb)]
x = rmsnorm(x)
```

兩種資訊相加，變成模型的輸入表示。

## Transformer 層的內部結構

每層 Transformer 包含兩個子層：

```
輸入 x
   │
   ├──→ SubLayer 1: Multi-Head Self-Attention ──+──→ 輸出 x'
   │                                               │
   └──→ SubLayer 2: Feed-Forward Network ─────────┘
```

### SubLayer 1：Multi-Head Self-Attention

讓每個位置能夠關注其他位置：

```python
# 計算 QKV
q = linear(x, W_q)
k = linear(x, W_k)
v = linear(x, W_v)

# KV Cache：用於加速推論
keys[li].append(k)
values[li].append(v)

# 多頭注意力
x_attn = []
for h in range(n_head):
    # 分割 QKV 到每個頭
    q_h = q[h*head_dim : (h+1)*head_dim]
    k_h = [ki[h*head_dim:(h+1)*head_dim] for ki in keys[li]]
    v_h = [vi[h*head_dim:(h+1)*head_dim] for vi in values[li]]
    
    # 注意力分數
    attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) 
                   for t in range(len(k_h))]
    
    # Softmax
    attn_weights = softmax(attn_logits)
    
    # 加權平均
    head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) 
               for j in range(head_dim)]
    x_attn.extend(head_out)

# 輸出投影
x_attn = linear(x_attn, W_o)

# 殘差連接
x = x + x_attn
```

### SubLayer 2：Feed-Forward Network

簡單的兩層網路，增加模型的表达能力：

```python
# 擴展維度
x = linear(x, W_fc1)      # 16 → 64
x = [xi.relu() for xi in x]  # ReLU 激活

# 收縮維度
x = linear(x, W_fc2)      # 64 → 16

# 殘差連接
x = x + x_residual
```

## RMSNorm：Transformer 的穩定器

microgpt 使用 RMSNorm（Root Mean Square Normalization）：

```python
def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)  # 均方值
    scale = (ms + 1e-5) ** -0.5             # 倒數平方根
    return [xi * scale for xi in x]
```

與 LayerNorm 的差異：只使用 RMS，不減均值。

## KV Cache：加速推論的技巧

生成文字時，每個新 token 都要重新計算整個序列的注意力。

KV Cache 把之前計算的 K 和 V 存起來：

```python
# 訓練時：每個位置獨立的 K, V
keys[li] = [k0, k1, k2, ...]  # 每步只有一個新的 k
values[li] = [v0, v1, v2, ...]  # 每步只有一個新的 v

# 推論時：重複使用之前的 K, V
# 只需要計算新的 Q，和累積的 K, V 做注意力
```

## 4. LM Head：預測下一個 Token

最後，用一個線性層把 hidden state 映射到詞彙表大小：

```python
logits = linear(x, state_dict['lm_head'])
# logits: [l0, l1, l2, ..., l26]（每個 token 的分數）
```

然後用 Softmax 轉換成機率分佈。

## 完整的資料流

```
輸入："alex" → tokens: [BOS, a, l, e, x]

Token 0 (BOS):
  Embedding + Position → x0
  Transformer Layer → h0
  LM Head → logits: [3.2, 1.5, ...]（預測 'a' 的分數最高）

Token 1 (a):
  Embedding + Position → x1
  Attention 看 [BOS] → context1
  MLP → h1
  LM Head → logits: [..., 4.1, ...]（預測 'l' 的分數最高）

... 繼續預測下一個字元
```

## 為什麼這個架構有效？

1. **Transformer**：任意位置可以直接關注其他位置
2. **Multi-Head**：學習多個不同的注意力模式
3. **殘差連接**：梯度流暢，訓練穩定
4. **RMSNorm**：數值穩定
5. **足夠多的參數**：學習複雜的模式

## 規模化

GPT-2 的架構（與 microgpt 相同）：

| 變數 | GPT-2 Small | microgpt |
|-----|------------|---------|
| n_layer | 12 | 1 |
| n_embd | 768 | 16 |
| n_head | 12 | 4 |
| vocab_size | 50257 | 27 |
| 參數量 | 117M | 4,192 |

只是把數字變大，演算法完全相同。

## 總結

這一章的核心概念：
- **任務**：下一個 token 預測
- **Embedding**：Token 和 Position 向量相加
- **Transformer 層**：Self-Attention + MLP + 殘差連接
- **LM Head**：把 hidden state 映射到詞彙表
- **KV Cache**：加速推論

下一章，我們看完整的訓練循環！

---

*上一步：[8. microgpt 完整解析](08-microgpt_overview.md)*  
*下一步：[10. 訓練循環與 Adam 優化器](10-training_loop.md)*
