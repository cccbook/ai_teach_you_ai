# 8. microgpt.py 完整解析——資料、Tokenizer、架構

## microgpt 是什麼？

microgpt 是 Andrej Karpathy 在 2026 年發布的一個極簡 GPT 實現，只有約 240 行 Python 程式碼，卻包含了完整 GPT 的所有核心元件。

原始碼：https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95

本書使用的程式碼位於：
- 模型：[gpt0.py](_code/nn0py/gpt0.py)
- 測試：[test_gpt0.py](_code/nn0py/test_gpt0.py)

## 五個必要元件

Karpathy 說：「我無法再簡化了。」microgpt 包含：

1. **資料集**：訓練用的文字
2. **Tokenizer**：把文字轉成數字
3. **Autograd**：自動微分引擎
4. **GPT 架構**：Transformer 模型
5. **優化器**：訓練參數更新

## 資料集：32,000 個名字

microgpt 用人名資料集訓練：

```python
# 下載名字列表
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/master/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')

# 讀取並 shuffle
docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")  # 32033
```

資料樣例：
```
alex, bob, charlie, david, emma, ...
```

目標：讓模型學會生成新的人名。

## Tokenizer：字元級別

microgpt 使用最簡單的字元級 Tokenizer：

```python
# 建立詞彙表
uchars = sorted(set(''.join(docs)))  # 所有唯一字元
BOS = len(uchars)  # 特殊 Begin of Sequence token
vocab_size = len(uchars) + 1

print(f"vocab size: {vocab_size}")  # 27（26 字母 + BOS）
```

詞彙表：
```
{'a': 0, 'b': 1, 'c': 2, ..., 'z': 25, 'BOS': 26}
```

## Autograd：與 micrograd 一樣

microgpt 使用與 micrograd 相同的 Value 類別實現自動微分：

```python
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')
    
    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))
    
    # ... 其他運算（relu, log, exp, pow, ...）
    
    def backward(self):
        # 反向傳播
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad
```

## 模型參數初始化

```python
n_layer = 1      # 1 層 Transformer
n_embd = 16      # Embedding 維度 16
block_size = 16  # 最大上下文長度
n_head = 4       # 4 個注意力頭
head_dim = n_embd // n_head  # 每頭 4 維

# 初始化權重矩陣（Gaussian random）
matrix = lambda nout, nin, std=0.08: [
    [Value(random.gauss(0, std)) for _ in range(nin)]
    for _ in range(nout)
]

# GPT 參數
state_dict = {
    'wte': matrix(vocab_size, n_embd),      # Token Embedding
    'wpe': matrix(block_size, n_embd),     # Position Embedding
    'lm_head': matrix(vocab_size, n_embd),  # Output Head
}

# 每層 Transformer 的參數
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)

params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"num params: {len(params)}")  # 4192 參數
```

## GPT 架構

完整的 `gpt()` 函數：

```python
def gpt(token_id, pos_id, keys, values):
    # 1. Embedding
    tok_emb = state_dict['wte'][token_id]  # Token embedding
    pos_emb = state_dict['wpe'][pos_id]    # Position embedding
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)
    
    # 2. Transformer 層
    for li in range(n_layer):
        x_residual = x
        x = rmsnorm(x)
        
        # 2.1 Self-Attention
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)
        
        # Multi-head attention
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            
            # 注意力分數 + Softmax
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) 
                          for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            
            # 加權平均
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) 
                       for j in range(head_dim)]
            x_attn.extend(head_out)
        
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]  # 殘差連接
        
        # 2.2 MLP
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]  # 殘差連接
    
    # 3. Output
    logits = linear(x, state_dict['lm_head'])
    return logits
```

## 輔助函數

### RMSNorm

Transformer 標準化的方法，比 LayerNorm 更簡單：

```python
def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]
```

### Linear 層

```python
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]
```

### Softmax

```python
def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]
```

## 參數數量分析

| 層面 | 計算 | 數量 |
|-----|------|------|
| Token Embedding | vocab_size × n_embd | 27 × 16 = 432 |
| Position Embedding | block_size × n_embd | 16 × 16 = 256 |
| Q, K, V | 3 × n_embd × n_embd | 3 × 256 = 768 |
| Attention Output | n_embd × n_embd | 256 |
| MLP fc1 | 4 × n_embd × n_embd | 4 × 256 = 1024 |
| MLP fc2 | n_embd × 4 × n_embd | 1024 |
| Output Head | vocab_size × n_embd | 432 |
| **總計** | | **4192** |

## 與 GPT-2 的規模差距

| 模型 | 參數量 | 差距 |
|-----|-------|------|
| microgpt | 4,192 | 1x |
| GPT-2 Small | 117M | ~28,000x |
| GPT-2 Medium | 345M | ~82,000x |
| GPT-2 Large | 774M | ~185,000x |
| GPT-3 | 175B | ~42,000,000x |

但核心演算法完全相同！

## 總結

這一章的核心概念：
- **資料集**：32,000 個名字
- **Tokenizer**：字元級，27 個 token
- **Autograd**：Value 類別實現的自動微分
- **GPT 架構**：Embedding → RMSNorm → Attention → MLP → RMSNorm → Output
- **參數量**：4,192 個

下一章，我們深入 GPT 架構的每個細節！

---

*上一步：[7. Self-Attention 機制解析](07-self_attention.md)*  
*下一步：[9. GPT 架構](09-gpt_architecture.md)*
