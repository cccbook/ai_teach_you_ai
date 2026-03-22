# 8. Complete Analysis of microgpt.py—Data, Tokenizer, Architecture

## What is microgpt?

microgpt is an ultra-minimal GPT implementation released by Andrej Karpathy in 2026, containing only about 240 lines of Python code, yet includes all core components of a complete GPT.

Source code: https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95

The code used in this book is located at:
- Model: [gpt0.py](../_code/nn0py/gpt0.py)
- Tests: [test_gpt0.py](../_code/nn0py/test_gpt0.py)

## Five Essential Components

Karpathy says: "I can't simplify this further." microgpt includes:

1. **Dataset**: text for training
2. **Tokenizer**: converts text to numbers
3. **Autograd**: automatic differentiation engine
4. **GPT Architecture**: Transformer model
5. **Optimizer**: training parameter updates

## Dataset: 32,000 Names

microgpt trains on a dataset of human names:

```python
# Download name list
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/master/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')

# Read and shuffle
docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")  # 32033
```

Data samples:
```
alex, bob, charlie, david, emma, ...
```

Goal: let the model learn to generate new human names.

## Tokenizer: Character-level

microgpt uses the simplest character-level Tokenizer:

```python
# Build vocabulary
uchars = sorted(set(''.join(docs)))  # All unique characters
BOS = len(uchars)  # Special Begin of Sequence token
vocab_size = len(uchars) + 1

print(f"vocab size: {vocab_size}")  # 27 (26 letters + BOS)
```

Vocabulary:
```
{'a': 0, 'b': 1, 'c': 2, ..., 'z': 25, 'BOS': 26}
```

## Autograd: Same as micrograd

microgpt implements automatic differentiation using the same Value class as micrograd:

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
    
    # ... other operations (relu, log, exp, pow, ...)
    
    def backward(self):
        # Backpropagation
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

## Model Parameter Initialization

```python
n_layer = 1      # 1 Transformer layer
n_embd = 16      # Embedding dimension 16
block_size = 16  # Maximum context length
n_head = 4       # 4 attention heads
head_dim = n_embd // n_head  # 4 dimensions per head

# Initialize weight matrices (Gaussian random)
matrix = lambda nout, nin, std=0.08: [
    [Value(random.gauss(0, std)) for _ in range(nin)]
    for _ in range(nout)
]

# GPT parameters
state_dict = {
    'wte': matrix(vocab_size, n_embd),      # Token Embedding
    'wpe': matrix(block_size, n_embd),     # Position Embedding
    'lm_head': matrix(vocab_size, n_embd),  # Output Head
}

# Parameters for each Transformer layer
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)

params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"num params: {len(params)}")  # 4192 parameters
```

## GPT Architecture

Complete `gpt()` function:

```python
def gpt(token_id, pos_id, keys, values):
    # 1. Embedding
    tok_emb = state_dict['wte'][token_id]  # Token embedding
    pos_emb = state_dict['wpe'][pos_id]    # Position embedding
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)
    
    # 2. Transformer layers
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
            
            # Attention scores + Softmax
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) 
                          for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            
            # Weighted average
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) 
                       for j in range(head_dim)]
            x_attn.extend(head_out)
        
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]  # Residual connection
        
        # 2.2 MLP
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]  # Residual connection
    
    # 3. Output
    logits = linear(x, state_dict['lm_head'])
    return logits
```

## Helper Functions

### RMSNorm

Transformer's normalization method, simpler than LayerNorm:

```python
def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]
```

### Linear Layer

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

## Parameter Count Analysis

| Layer | Calculation | Count |
|-------|-------------|-------|
| Token Embedding | vocab_size × n_embd | 27 × 16 = 432 |
| Position Embedding | block_size × n_embd | 16 × 16 = 256 |
| Q, K, V | 3 × n_embd × n_embd | 3 × 256 = 768 |
| Attention Output | n_embd × n_embd | 256 |
| MLP fc1 | 4 × n_embd × n_embd | 4 × 256 = 1024 |
| MLP fc2 | n_embd × 4 × n_embd | 1024 |
| Output Head | vocab_size × n_embd | 432 |
| **Total** | | **4192** |

## Scale Comparison with GPT-2

| Model | Parameters | Gap |
|-------|------------|-----|
| microgpt | 4,192 | 1x |
| GPT-2 Small | 117M | ~28,000x |
| GPT-2 Medium | 345M | ~82,000x |
| GPT-2 Large | 774M | ~185,000x |
| GPT-3 | 175B | ~42,000,000x |

But the core algorithm is exactly the same!

## Summary

Core concepts from this chapter:
- **Dataset**: 32,000 names
- **Tokenizer**: character-level, 27 tokens
- **Autograd**: automatic differentiation with Value class
- **GPT Architecture**: Embedding → RMSNorm → Attention → MLP → RMSNorm → Output
- **Parameter count**: 4,192

In the next chapter, we dive deep into every detail of the GPT architecture!

---

*Previous: [7. Self-Attention Mechanism Explained](07-self_attention.md)*  
*Next: [9. GPT Architecture](09-gpt_architecture.md)*
