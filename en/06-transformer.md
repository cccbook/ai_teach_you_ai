# 6. Transformer—The Art of Attention

## From Sequence to Sequence

There's a fundamental problem with processing text: **order matters**.

"Dog bites man" and "man bites dog" consist of the same characters but have completely different meanings.

Traditional methods (RNN, LSTM) process sequentially:
```
Input: dog → bites → man
  RNN:  h1    h2    h3 (each step considers previous context)
```

Problem: When the distance for information transmission is too long, it decays. The influence of "dog" becomes fuzzy by the time it reaches "man".

## The Transformer Revolution

In 2017, Google published "Attention Is All You Need", introducing the Transformer:

**Core idea: replace sequential processing with attention mechanism.**

Instead of processing in order, let every position **directly attend to** all other positions.

## Architecture Overview

Transformer (using Encoder-Decoder as example):

```
                     Output
                       │
                    Linear
                       │
                    Softmax
                       │
                  LayerNorm
                       │
                     FFN ←─── Residual Connection
                       │
                  LayerNorm
                       │
                 Self-Attention ←─── Residual Connection
                       │
                     Output Embedding
                       │
                    Positional Encoding
                       │
                     Output Tokens
                       │
                       │  (Decoder autoregressive generation)
                       │
                     Output Embedding
                       │
                  LayerNorm
                       │
                    Cross-Attention ←─── Residual Connection (attending to Encoder)
                       │
                  LayerNorm
                       │
                 Self-Attention ←─── Residual Connection
                       │
                    Linear
                       │
                    Softmax
                       │
                  LayerNorm
                       │
                     FFN ←─── Residual Connection
                       │
                  LayerNorm
                       │
                 Self-Attention ←─── Residual Connection
                       │
                  Input Embedding
                       │
                    Positional Encoding
                       │
                     Input Tokens
```

## Key Component: Multi-Head Self-Attention

### Self-Attention

Self-Attention allows each position in a sequence to "attend to" other positions:

```
"dog"   attends to: itself, bites, man
"bites" attends to: dog, itself, man
"man"   attends to: dog, bites, itself
```

Mathematically implemented through Query, Key, Value:

$$Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### Multi-Head Attention

Multi-head attention: use multiple QKV sets to attend to different aspects simultaneously:

```
Input → Multi-Head Attention → Concatenate Output
   │
   ├── Head 1: syntactic relationships
   ├── Head 2: semantic relationships
   ├── Head 3: positional relationships
   └── ...
```

```python
class MultiHeadAttention:
    def __init__(self, d_model, n_heads):
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Initialize QKV projection matrices
        self.W_q = matrix(d_model, d_model)
        self.W_k = matrix(d_model, d_model)
        self.W_v = matrix(d_model, d_model)
        self.W_o = matrix(d_model, d_model)
```

### Why Multiple Heads?

A single head might only learn one pattern. Multi-head allows the model to learn multiple relationships simultaneously.

## Feed-Forward Network (FFN)

After each attention layer, there's a simple feed-forward network:

```python
def ffn(x):
    x = linear(x, W_fc1)  # Expand dimensions: d_model → 4*d_model
    x = relu(x)
    x = linear(x, W_fc2)  # Contract dimensions: 4*d_model → d_model
    return x
```

Seems simple, yet it accounts for most of the Transformer's parameters.

## LayerNorm and Residual Connections

### LayerNorm

Normalize each layer's output to help stabilize training:

$$LayerNorm(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

### Residual Connections

Directly add input to output to alleviate gradient problems:

```python
x = x + attention(x)  # Residual connection
x = x + ffn(x)       # Residual connection
```

## GPT Series Transformer: Decoder Only

GPT (Generative Pre-trained Transformer) uses only the Decoder portion:

```
Input: I like to
Output: eat
```

Characteristics:
- **Masked Self-Attention**: each position can only see itself and previous content
- **Unidirectional**: generate left to right

```
                     Output: eat
                       │
                    Linear
                       │
                  LayerNorm
                       │
                     FFN
                       │
                  LayerNorm
                       │
           Masked Self-Attention (only sees I like to)
                       │
                    Token Embedding
                       │
                    Positional Encoding
                       │
                   I  like  to
```

## Transformer vs RNN/LSTM

| Feature | Transformer | RNN/LSTM |
|---------|------------|----------|
| Parallelism | Fully parallel | Sequential processing |
| Long-range dependencies | O(1) attend to any position | O(n), decays |
| Training speed | Fast (GPU-friendly) | Slow |
| Inference speed | Slow (autoregressive generation) | Fast |
| Memory capacity | Fixed (limited by positional encoding) | Variable |

## Scaling: Bigger is Stronger

Transformer's scaling laws:

```
More parameters ↑ = More capability ↑
More training data ↑ = More capability ↑
Both ↑ = Capability ↑↑↑
```

| Model | Parameters |
|-------|------------|
| microgpt | ~4,000 |
| GPT-2 small | 117M |
| GPT-3 | 175B |
| GPT-4 | Estimated 1-2T |

## Summary

Core concepts from this chapter:
- **Transformer**: revolutionary architecture replacing sequential processing with attention
- **Self-Attention**: allows each position to attend to all other positions
- **QKV mechanism**: Query, Key, Value projections and attention computation
- **Multi-Head**: learning multiple relationship patterns simultaneously
- **FFN + LayerNorm + Residual**: key modules for stable training

In the next chapter, we dive deep into the mathematical details of Self-Attention!

---

*Previous: [5. Tokenization and Embedding](05-embedding.md)*  
*Next: [7. Self-Attention Mechanism Explained](07-self_attention.md)*
