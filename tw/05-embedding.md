# 5. 文字的數學——Token 化與 Embedding

## 電腦如何理解文字？

前幾章我們處理的是數字。但 AI 要處理的是文字——圖片、聲音、程式碼都可以轉換成文字描述。

問題來了：**電腦怎麼「理解」文字？**

答案是：**把文字轉換成數字。**

## Token 化：文字 → 數字序列

Token 化是將文字切成小塊，每小塊給一個數字 ID。

### 簡單的詞彙表

```python
vocab = {
    "我": 0,
    "愛": 1,
    "你": 2,
    "學": 3,
    "習": 4
}

text = "我愛學習"
tokens = [vocab[c] for c in text]
print(tokens)  # [0, 1, 3, 4]
```

### 字元級 vs 詞彙級

| 方式 | 優點 | 缺點 |
|-----|------|------|
| 字元級 | 詞彙表小，處理新字靈活 | token 序列變長 |
| 詞彙級 | 序列短，語義更明確 | 詞彙表大，難處理新詞 |

### BPE：兩者的平衡

BPE（Byte Pair Encoding）是現代模型常用的方法：
- 把常見的字元組合當成一個 token
- 兼顧詞彙表大小和序列長度
- 能處理從未見過的單詞（拆成已知子詞）

GPT-4 使用的 tiktoken 就是 BPE 的一種實現。

### microgpt 的Tokenizer

microgpt 使用最簡單的**字元級Tokenizer**：

```python
# 從名字列表建立詞彙表
docs = ["alex", "bob", "charlie", ...]
chars = sorted(set(''.join(docs)))  # 所有唯一字元
vocab = {c: i for i, c in enumerate(chars)}

# vocab = {'a': 0, 'b': 1, 'c': 2, ..., 'z': 25, 'eos': 26}
```

## Embedding：數字 → 向量

Tokenizer 給每個 token 一個 ID，但我們需要更多資訊。

問題：ID = 0 的「a」和 ID = 1 的「b」，它們的數字關係是什麼？

**沒有意義！** 字母表順序不代表語意相似度。

### 解決方案：Embedding

把每個 token 映射到一個**向量**（一串數字）：

```
"cat"  → [0.2, -0.5, 0.8, ...]  (300 維向量)
"dog"  → [0.3, -0.4, 0.7, ...]  (相似！)
"car"  → [0.8, 0.2, -0.3, ...]  (不相關)
```

### Embedding 是怎麼來的？

一開始是隨機的，訓練過程中學到的：
1. 初始化：隨機向量
2. 訓練：根據任務調整向量
3. 結果：相似語意的詞彙會有相似的向量

## 向量空間：語意的幾何

Embedding 把語意向量化，語意相似的事物在向量空間中靠近。

### 類比推理

最著名的例子：

```
國王 - 男人 + 女人 ≈ 女王
```

向量運算：

```python
king = embedding["king"]
man = embedding["man"]
woman = embedding["woman"]

queen ≈ king - man + woman
```

### 相似度：點積

衡量兩個向量有多相似，用**點積**（Dot Product）：

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# "cat" 和 "dog" 相似
similarity("cat", "dog") = 0.87

# "cat" 和 "car" 不太相似
similarity("cat", "car") = 0.23
```

## Positional Encoding：位置資訊

文字有順序，但神經網路一開始不知道位置。

**位置編碼**（Positional Encoding）把位置也轉成向量：

```
第 0 個 token → [1, 0, 0, ...]
第 1 個 token → [0, 1, 0, ...]
第 2 個 token → [0, 0, 1, ...]
```

Transformer 用正弦/餘弦函數產生位置編碼：

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$$

## Transformer 的輸入處理

完整的 Transformer 輸入流程：

```
文字 → Token IDs → Embedding 向量 → 加上位置編碼 → 輸入 Transformer
```

```python
# 示意代碼
token_ids = tokenizer("Hello world")  # [0, 1]
token_embeddings = embedding(token_ids)  # [[0.2, ...], [0.5, ...]]
position_embeddings = positional_encoding(2)
input_vectors = token_embeddings + position_embeddings  # 加入位置資訊
```

## 視覺化：詞彙的地圖

用降維技術（如 t-SNE、PCA）可以把高維 Embedding 投影到 2D：

```
                    動物
                   ╱    ╲
                貓 ── 狗   交通工具
                 ╲        ╱  ╲
                  bird  car
                       ╱
                   顏色
                   ╱
                red - blue - green
```

相似的詞彙聚集在一起。

## Embedding 的維度

| 任務 | 維度 | 說明 |
|-----|------|------|
| 小型實驗 | 16-64 | micrograd/microgpt |
| 小型模型 | 128-512 | GPT-2 small |
| 中型模型 | 768-1024 | GPT-2 medium/large |
| 大型模型 | 4096+ | GPT-4 等 |

維度越高，能表達的語意越細緻，但計算成本也越高。

## 總結

這一章的核心概念：
- **Token 化**：把文字切成小塊，給予數字 ID
- **Embedding**：把 token ID 轉換成有意義的向量
- **位置編碼**：加入順序資訊
- **向量空間**：語意相似的事物在空間中靠近

下一章，我們進入 Transformer 的核心——注意力機制！

---

*上一步：[4. 用 micrograd 理解自動微分](04-micrograd.md)*  
*下一步：[6. Transformer——注意力的藝術](06-transformer.md)*
