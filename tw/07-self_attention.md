# 7. Self-Attention 機制解析

## 為什麼叫「Self」Attention？

「Self」是因為這裡的 QKV 都來自同一個輸入。

對比：
- **Self-Attention**：輸入關注輸入自己
- **Cross-Attention**：解碼器關注編碼器（翻譯任務）

## QKV：三個視角看同一個內容

每個 token 有三個向量，代表三種不同的「視角」：

| 向量 | 意義 | 比喻 |
|-----|------|-----|
| Query（查詢） | 我在找什麼 | 搜尋引擎的搜尋框 |
| Key（鑰匙） | 我有什麼 | 網頁的標題 |
| Value（數值） | 我的實際內容 | 網頁的內容 |

### QKV 的產生

```python
# 輸入向量 x（假設 d_model = 4）
x = [0.5, -0.2, 0.8, 0.1]

# 通過權重矩陣產生 QKV
q = x @ W_q  # [q1, q2, q3, q4]
k = x @ W_k  # [k1, k2, k3, k4]
v = x @ W_v  # [v1, v2, v3, v4]
```

## 注意力分數：Query 和 Key 的匹配

「狗」想知道應該關注「什麼」，比較自己的 Query 和所有 token 的 Key：

```
Token:  狗    咬    人
Query: [0.5, -0.2, 0.8, 0.1]  ← 「狗」的 Query

Key:    [0.3, -0.1, 0.7, 0.2]  ← 「狗」的 Key
        [0.1, 0.4, -0.3, 0.5]  ← 「咬」的 Key
        [0.2, -0.5, 0.2, 0.3]  ← 「人」的 Key
```

計算點積（相似度）：
```python
score_狗_狗 = dot(q_狗, k_狗)  # 狗關注自己
score_狗_咬 = dot(q_狗, k_咬)  # 狗關注咬
score_狗_人 = dot(q_狗, k_人)  # 狗關注人
```

## Softmax：轉換成機率

把分數通過 Softmax，變成「關注權重」：

$$softmax(s) = \frac{e^s}{\sum e^s}$$

```
分數：[3.2, 1.5, -0.8]
Softmax：[0.82, 0.17, 0.01]

「狗」這個 token：
- 82% 關注自己
- 17% 關注「咬」
- 1% 關注「人」
```

## 注意力輸出：加權平均 Value

用關注權重，對所有 Value 做加權平均：

```python
output = 0.82 * v_狗 + 0.17 * v_咬 + 0.01 * v_人
```

直覺：把其他 token 的資訊「濃縮」到當前 token。

## 完整公式

$$Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中 $\sqrt{d_k}$ 是為了防止點積過大導致 Softmax 梯度消失。

## Masked Self-Attention：只看向過去

GPT 是生成模型，不能「偷看」未來的 token。

用一個 Mask 把未來的分數設為 -∞：

```python
def masked_attention(q, k, v):
    # 計算注意力分數
    scores = q @ k.T
    
    # 應用 Mask（上三角設為 -inf）
    seq_len = len(q)
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1) * -1e9
    scores = scores + mask
    
    # Softmax
    weights = softmax(scores)
    
    # 加權平均
    return weights @ v
```

視覺化：
```
分數矩陣（i 行 j 列 = token i 關注 token j）：

     j:0  1  2  3
   ┌──────────────
i:0│  1  0  0  0    ← 第 0 個 token 只能看自己
i:1│  1  1  0  0    ← 第 1 個 token 只能看 0, 1
i:2│  1  1  1  0    ← 第 2 個 token 只能看 0, 1, 2
i:3│  1  1  1  1    ← 第 3 個 token 可以看全部
```

## Multi-Head Attention

把 QKV 分成多個「頭」，每個頭獨立的 QKV：

```python
class MultiHeadAttention:
    def __init__(self, d_model=64, n_heads=4):
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads  # 16
        
        # 每個頭有自己的權重
        self.W_q = [matrix(self.head_dim, d_model) for _ in range(n_heads)]
        self.W_k = [matrix(self.head_dim, d_model) for _ in range(n_heads)]
        self.W_v = [matrix(self.head_dim, d_model) for _ in range(n_heads)]
        self.W_o = matrix(d_model, d_model)
    
    def forward(self, x):
        outputs = []
        for h in range(self.n_heads):
            q = x @ self.W_q[h]
            k = x @ self.W_k[h]
            v = x @ self.W_v[h]
            head_out = attention(q, k, v)  # 單頭注意力
            outputs.append(head_out)
        
        # 拼接所有頭的輸出
        concat = concat(outputs, dim=-1)
        
        # 最後一次線性投影
        return concat @ self.W_o
```

## microgpt 的實現

microgpt 用純 Python 實現了 Multi-Head Attention：

```python
def gpt(token_id, pos_id, keys, values):
    # ... embedding ...
    
    for li in range(n_layer):
        # Self-Attention
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        
        # 儲存 KV cache（用於加速推論）
        keys[li].append(k)
        values[li].append(v)
        
        # 多頭計算
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            
            # 注意力分數
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) 
                           for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            
            # 加權平均
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) 
                       for j in range(head_dim)]
            x_attn.extend(head_out)
        
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
```

## 為什麼 Attention 這麼有效？

1. **直接連接**：任意兩個位置之間可以直接計算關注分數
2. **動態權重**：根據輸入動態調整關注模式
3. **可解釋性**：可視化注意力權重，看模型在「想」什麼

## 總結

這一章的核心概念：
- **QKV 機制**：Query 找、Key 匹配、Value 提供資訊
- **注意力分數**：Query 和 Key 的點積
- **Softmax**：分數轉換為機率分佈
- **加權平均**：用注意力權重組合 Value
- **Masked Attention**：防止看到未來資訊
- **Multi-Head**：同時關注多個模式

下一章，我們用 microgpt 把所有概念串聯起來！

---

*上一步：[6. Transformer](06-transformer.md)*  
*下一步：[8. microgpt.py 完整解析](08-microgpt_overview.md)*
