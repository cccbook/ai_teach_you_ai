# 7. Self-Attention Mechanism Explained

## Why Is It Called "Self" Attention?

It's called "Self" because the QKV here all come from the same input.

Comparison:
- **Self-Attention**: input attends to itself
- **Cross-Attention**: decoder attends to encoder (for translation tasks)

## QKV: Three Perspectives on the Same Content

Each token has three vectors, representing three different "perspectives":

| Vector | Meaning | Analogy |
|--------|---------|---------|
| Query | What am I looking for? | Search engine search box |
| Key | What do I have? | Web page title |
| Value | My actual content | Web page content |

### Generating QKV

```python
# Input vector x (假设 d_model = 4)
x = [0.5, -0.2, 0.8, 0.1]

# Generate QKV through weight matrices
q = x @ W_q  # [q1, q2, q3, q4]
k = x @ W_k  # [k1, k2, k3, k4]
v = x @ W_v  # [v1, v2, v3, v4]
```

## Attention Scores: Query and Key Matching

"Dog" wants to know what to "attend to", comparing its Query with all tokens' Keys:

```
Token:  dog   bites  man
Query: [0.5, -0.2, 0.8, 0.1]  ← "dog"'s Query

Key:    [0.3, -0.1, 0.7, 0.2]  ← "dog"'s Key
        [0.1, 0.4, -0.3, 0.5]  ← "bites"'s Key
        [0.2, -0.5, 0.2, 0.3]  ← "man"'s Key
```

Compute dot product (similarity):
```python
score_dog_dog = dot(q_dog, k_dog)  # dog attends to itself
score_dog_bites = dot(q_dog, k_bites)  # dog attends to bites
score_dog_man = dot(q_dog, k_man)  # dog attends to man
```

## Softmax: Converting to Probabilities

Pass scores through Softmax to become "attention weights":

$$softmax(s) = \frac{e^s}{\sum e^s}$$

```
Scores: [3.2, 1.5, -0.8]
Softmax: [0.82, 0.17, 0.01]

Token "dog":
- 82% attends to itself
- 17% attends to "bites"
- 1% attends to "man"
```

## Attention Output: Weighted Average of Values

Use attention weights to compute weighted average of all Values:

```python
output = 0.82 * v_dog + 0.17 * v_bites + 0.01 * v_man
```

Intuition: "condense" information from other tokens into the current token.

## Complete Formula

$$Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where $\sqrt{d_k}$ prevents dot products from becoming too large, causing Softmax gradients to vanish.

## Masked Self-Attention: Looking Only at the Past

GPT is a generative model and cannot "peek" at future tokens.

Use a Mask to set future scores to -∞:

```python
def masked_attention(q, k, v):
    # Compute attention scores
    scores = q @ k.T
    
    # Apply Mask (set upper triangle to -inf)
    seq_len = len(q)
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1) * -1e9
    scores = scores + mask
    
    # Softmax
    weights = softmax(scores)
    
    # Weighted average
    return weights @ v
```

Visualization:
```
Score matrix (row i, column j = token i attends to token j):

     j:0  1  2  3
   ┌──────────────
i:0│  1  0  0  0    ← Token 0 can only see itself
i:1│  1  1  0  0    ← Token 1 can only see 0, 1
i:2│  1  1  1  0    ← Token 2 can only see 0, 1, 2
i:3│  1  1  1  1    ← Token 3 can see all
```

## Multi-Head Attention

Split QKV into multiple "heads", each with independent QKV:

```python
class MultiHeadAttention:
    def __init__(self, d_model=64, n_heads=4):
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads  # 16
        
        # Each head has its own weights
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
            head_out = attention(q, k, v)  # Single-head attention
            outputs.append(head_out)
        
        # Concatenate all heads' outputs
        concat = concat(outputs, dim=-1)
        
        # Final linear projection
        return concat @ self.W_o
```

## microgpt Implementation

microgpt implements Multi-Head Attention in pure Python:

```python
def gpt(token_id, pos_id, keys, values):
    # ... embedding ...
    
    for li in range(n_layer):
        # Self-Attention
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        
        # Store KV cache (for faster inference)
        keys[li].append(k)
        values[li].append(v)
        
        # Multi-head computation
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            
            # Attention scores
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) 
                           for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            
            # Weighted average
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) 
                       for j in range(head_dim)]
            x_attn.extend(head_out)
        
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
```

## Why Is Attention So Effective?

1. **Direct connections**: any two positions can directly compute attention scores
2. **Dynamic weights**: attention patterns dynamically adjust based on input
3. **Interpretability**: visualize attention weights to see what the model is "thinking"

## Summary

Core concepts from this chapter:
- **QKV mechanism**: Query finds, Key matches, Value provides information
- **Attention scores**: dot product of Query and Key
- **Softmax**: converts scores to probability distribution
- **Weighted average**: combine Values using attention weights
- **Masked Attention**: prevents seeing future information
- **Multi-Head**: attend to multiple patterns simultaneously

In the next chapter, we connect all concepts together with microgpt!

---

*Previous: [6. Transformer](06-transformer.md)*  
*Next: [8. Complete Analysis of microgpt.py](08-microgpt_overview.md)*
