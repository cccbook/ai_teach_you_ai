# 9. GPT Architecture: Embedding → Transformer Layers → LM Head

## GPT's Task: Next Token Prediction

GPT (Generative Pre-trained Transformer) has a very simple task:

**Given all previous tokens, predict the next token.**

```
Input: I like to
Output: eat

Input: The sky is
Output: blue
```

During training, the model learns this mapping.

## Three Main Stages

```
Token IDs → Token Embedding → + Position Embedding → Transformer Layers → LM Head → Logits
                     ↑                                              ↓
               Lookup                                          Predict next Token
```

## 1. Token Embedding

Convert token ID to a vector:

```python
# state_dict['wte']: vocab_size × n_embd matrix
tok_emb = state_dict['wte'][token_id]  # Get the token_id-th row vector
# tok_emb: [v0, v1, v2, ..., v15] (16-dimensional vector)
```

Intuition: Each character has its own "personality vector".

## 2. Position Embedding

Let the model know the position of each token:

```python
# state_dict['wpe']: block_size × n_embd matrix
pos_emb = state_dict['wpe'][pos_id]  # Get the pos_id-th row vector
```

## 3. Fusion: Token + Position

```python
x = [t + p for t, p in zip(tok_emb, pos_emb)]
x = rmsnorm(x)
```

Combine both pieces of information into the model's input representation.

## Transformer Layer Internal Structure

Each Transformer layer contains two sublayers:

```
Input x
   │
   ├──→ SubLayer 1: Multi-Head Self-Attention ──+──→ Output x'
   │                                               │
   └──→ SubLayer 2: Feed-Forward Network ─────────┘
```

### SubLayer 1: Multi-Head Self-Attention

Allows each position to attend to other positions:

```python
# Compute QKV
q = linear(x, W_q)
k = linear(x, W_k)
v = linear(x, W_v)

# KV Cache: Used for faster inference
keys[li].append(k)
values[li].append(v)

# Multi-head attention
x_attn = []
for h in range(n_head):
    # Split QKV to each head
    q_h = q[h*head_dim : (h+1)*head_dim]
    k_h = [ki[h*head_dim:(h+1)*head_dim] for ki in keys[li]]
    v_h = [vi[h*head_dim:(h+1)*head_dim] for vi in values[li]]
    
    # Attention scores
    attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) 
                   for t in range(len(k_h))]
    
    # Softmax
    attn_weights = softmax(attn_logits)
    
    # Weighted average
    head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) 
               for j in range(head_dim)]
    x_attn.extend(head_out)

# Output projection
x_attn = linear(x_attn, W_o)

# Residual connection
x = x + x_attn
```

### SubLayer 2: Feed-Forward Network

A simple two-layer network that increases the model's expressiveness:

```python
# Expand dimensions
x = linear(x, W_fc1)      # 16 → 64
x = [xi.relu() for xi in x]  # ReLU activation

# Contract dimensions
x = linear(x, W_fc2)      # 64 → 16

# Residual connection
x = x + x_residual
```

## RMSNorm: The Stabilizer for Transformers

microgpt uses RMSNorm (Root Mean Square Normalization):

```python
def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)  # Mean square
    scale = (ms + 1e-5) ** -0.5             # Inverse square root
    return [xi * scale for xi in x]
```

Difference from LayerNorm: Only uses RMS, no mean subtraction.

## KV Cache: Technique for Faster Inference

When generating text, each new token requires recomputing attention over the entire sequence.

KV Cache stores previously computed K and V:

```python
# During training: Independent K, V for each position
keys[li] = [k0, k1, k2, ...]  # Only one new k per step
values[li] = [v0, v1, v2, ...]  # Only one new v per step

# During inference: Reuse previous K, V
# Only need to compute new Q, and attend with accumulated K, V
```

## 4. LM Head: Predict Next Token

Finally, use a linear layer to map hidden state to vocabulary size:

```python
logits = linear(x, state_dict['lm_head'])
# logits: [l0, l1, l2, ..., l26] (scores for each token)
```

Then use Softmax to convert to probability distribution.

## Complete Data Flow

```
Input: "alex" → tokens: [BOS, a, l, e, x]

Token 0 (BOS):
  Embedding + Position → x0
  Transformer Layer → h0
  LM Head → logits: [3.2, 1.5, ...] (highest score for 'a')

Token 1 (a):
  Embedding + Position → x1
  Attention sees [BOS] → context1
  MLP → h1
  LM Head → logits: [..., 4.1, ...] (highest score for 'l')

... continue predicting next character
```

## Why Does This Architecture Work?

1. **Transformer**: Any position can directly attend to other positions
2. **Multi-Head**: Learn multiple different attention patterns
3. **Residual Connections**: Smooth gradient flow, stable training
4. **RMSNorm**: Numerical stability
5. **Enough Parameters**: Learn complex patterns

## Scaling

GPT-2 architecture (same as microgpt):

| Variable | GPT-2 Small | microgpt |
|---------|------------|---------|
| n_layer | 12 | 1 |
| n_embd | 768 | 16 |
| n_head | 12 | 4 |
| vocab_size | 50257 | 27 |
| Parameters | 117M | 4,192 |

Just scale up the numbers, the algorithm is exactly the same.

## Summary

Key concepts from this chapter:
- **Task**: Next token prediction
- **Embedding**: Token and Position vectors are added together
- **Transformer Layers**: Self-Attention + MLP + Residual Connection
- **LM Head**: Map hidden state to vocabulary
- **KV Cache**: Accelerate inference

In the next chapter, we look at the complete training loop!

---

*Previous: [8. microgpt Complete Analysis](08-microgpt_overview.md)*  
*Next: [10. Training Loop and Adam Optimizer](10-training_loop.md)*
