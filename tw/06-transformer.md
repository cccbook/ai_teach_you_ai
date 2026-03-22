# 6. Transformer——注意力的藝術

## 從序列到序列

處理文字有個根本問題：**順序很重要**。

「狗咬人」和「人咬狗」由同樣的字組成，意思完全不同。

傳統方法（RNN、LSTM）按順序處理：
```
輸入：狗 → 咬 → 人
 RNN： h1   h2   h3（每步都考慮之前的 context）
```

問題：資訊傳遞距離太長時會衰減，「狗」の影響到「人」時已經模糊了。

## Transformer 的革命

2017 年，Google 發表了「Attention Is All You Need」論文，，提出了 Transformer：

**核心思想：用注意力機制取代序列處理。**

不再按順序處理，而是讓每個位置都能**直接關注**其他所有位置。

## 架構總覽

Transformer（以 Encoder-Decoder 為例）：

```
                    輸出
                      │
                   Linear
                      │
                   Softmax
                      │
                 LayerNorm
                      │
                    FFN ←─── 殘差連接
                      │
                 LayerNorm
                      │
                Self-Attention ←─── 殘差連接
                      │
                    Output Embedding
                      │
                   Positional Encoding
                      │
                    Output Tokens
                      │
                      │  （解碼器自迴歸生成）
                      │
                    Output Embedding
                      │
                 LayerNorm
                      │
                   Cross-Attention ←─── 殘差連接（看向 Encoder）
                      │
                 LayerNorm
                      │
                Self-Attention ←─── 殘差連接
                      │
                   Linear
                      │
                   Softmax
                      │
                 LayerNorm
                      │
                    FFN ←─── 殘差連接
                      │
                 LayerNorm
                      │
                Self-Attention ←─── 殘差連接
                      │
                 Input Embedding
                      │
                   Positional Encoding
                      │
                    Input Tokens
```

## 關鍵組成：Multi-Head Self-Attention

### Self-Attention

Self-Attention（自注意力）讓序列中的每個位置去「注意」其他位置：

```
"狗" 關注：自己、咬、人
"咬" 關注：狗、自己、人
"人" 關注：狗、咬、自己
```

數學上，通過 Query、Key、Value 實現：

$$Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### Multi-Head Attention

多頭注意力：用多組 QKV，同時關注不同方面：

```
輸入 → 多頭注意力 → 拼接輸出
   │
   ├── Head 1: 語法關係
   ├── Head 2: 語意關係
   ├── Head 3: 位置關係
   └── ...
```

```python
class MultiHeadAttention:
    def __init__(self, d_model, n_heads):
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # 初始化 QKV 投影矩陣
        self.W_q = matrix(d_model, d_model)
        self.W_k = matrix(d_model, d_model)
        self.W_v = matrix(d_model, d_model)
        self.W_o = matrix(d_model, d_model)
```

### 為什麼多頭？

一個頭可能只學到一種模式。多頭讓模型同時學習多種關係。

## Feed-Forward Network（FFN）

每層注意力之後，有一個簡單的前饋網路：

```python
def ffn(x):
    x = linear(x, W_fc1)  # 擴展維度：d_model → 4*d_model
    x = relu(x)
    x = linear(x, W_fc2)  # 收縮維度：4*d_model → d_model
    return x
```

看似簡單，卻佔了 Transformer 參數量的大部分。

## LayerNorm 與殘差連接

### LayerNorm

標準化每層的輸出，幫助訓練穩定：

$$LayerNorm(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

### 殘差連接

把輸入直接加到輸出上，緩解梯度問題：

```python
x = x + attention(x)  # 殘差連接
x = x + ffn(x)       # 殘差連接
```

## GPT 系列的 Transformer：只有 Decoder

GPT（Generative Pre-trained Transformer）只使用 Decoder 部分：

```
輸入：I like to
輸出：eat
```

特點：
- **Masked Self-Attention**：每個位置只能看到自己和之前的內容
- **單向**：從左到右生成

```
                    輸出 eat
                      │
                   Linear
                      │
                 LayerNorm
                      │
                    FFN
                      │
                 LayerNorm
                      │
          Masked Self-Attention（只看 I like to）
                      │
                   Token Embedding
                      │
                   Positional Encoding
                      │
                  I  like  to
```

## Transformer vs RNN/LSTM

| 特性 | Transformer | RNN/LSTM |
|-----|------------|---------|
| 並行性 | 完全並行 | 順序處理 |
| 長距離依賴 | O(1) 關注任意位置 | O(n)，會衰減 |
| 訓練速度 | 快（GPU 友善） | 慢 |
| 推論速度 | 慢（自迴歸生成） | 快 |
| 記憶�量 | 固定（位置編碼限制） | 可變 |

## 規模化：越大越強

Transformer 的規模法則：

```
參數量 ↑ = 能力 ↑
訓練資料 ↑ = 能力 ↑
兩者都 ↑ = 能力 ↑↑↑
```

| 模型 | 參數量 |
|-----|-------|
| microgpt | ~4,000 |
| GPT-2 small | 117M |
| GPT-3 | 175B |
| GPT-4 | 估計 1-2T |

## 總結

這一章的核心概念：
- **Transformer**：用注意力取代序列處理的革命性架構
- **Self-Attention**：讓每個位置關注其他所有位置
- **QKV 機制**：Query、Key、Value 的投影與注意力計算
- **Multi-Head**：同時學習多種關係模式
- **FFN + LayerNorm + 殘差**：穩定訓練的關鍵模組

下一章，我們深入 Self-Attention 的數學細節！

---

*上一步：[5. Token 化與 Embedding](05-embedding.md)*  
*下一步：[7. Self-Attention 機制解析](07-self_attention.md)*
