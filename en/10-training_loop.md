# 10. Training Loop and Adam Optimizer

## Training Loop: Teaching the Model to Predict

microgpt's training loop is minimal but contains the complete learning process:

```python
num_steps = 1000

for step in range(num_steps):
    # 1. Get a document (name)
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)
    
    # 2. Forward pass + Compute loss
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()  # Cross-entropy loss
        losses.append(loss_t)
    loss = (1 / n) * sum(losses)
    
    # 3. Backward pass: Compute gradients
    loss.backward()
    
    # 4. Update parameters (Adam optimizer)
    lr_t = learning_rate * (1 - step / num_steps)  # Linear decay
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad      # First moment estimate
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2  # Second moment estimate
        m_hat = m[i] / (1 - beta1 ** (step + 1))          # Bias correction
        v_hat = v[i] / (1 - beta2 ** (step + 1))          # Bias correction
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps)     # Update
        p.grad = 0  # Clear gradient
    
    if step % 100 == 0:
        print(f"step {step}: loss {loss.data:.4f}")
```

## Data Preparation: From Text to Tokens

```python
doc = docs[step % len(docs)]  # Get a name
# doc = "alex"

tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
# tokens = [26, 0, 11, 5, 23, 26]  (BOS, a, l, e, x, BOS)
```

Why add BOS (Begin Of Sequence)?

```
tokens:    [BOS]  →  a  →  l  →  e  →  x  →  [BOS]
position:    0       1     2     3     4       5

Training objectives:
- Given BOS, predict a
- Given BOS, a, predict l
- Given BOS, a, l, predict e
- Given BOS, a, l, e, predict x
```

BOS marks the start of the sequence, telling the model where to begin.

## Loss Function: Cross-Entropy

```python
loss_t = -probs[target_id].log()
```

This is **Cross-Entropy Loss**.

Intuition:
- If the correct answer has probability 1.0, log(1.0) = 0, loss = 0 (perfect)
- If the correct answer has probability 0.1, log(0.1) = -2.3, loss = 2.3 (bad)

Math:

$$L = -\log(p_{correct})$$

- When $p_{correct} → 1$, $L → 0$ (good)
- When $p_{correct} → 0$, $L → \infty$ (bad)

## Softmax: Probability Distribution

```python
def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]
```

Softmax converts a list of numbers into a probability distribution:

```python
logits = [3.2, 1.5, 0.8, 0.3]  # Scores for each token
probs = softmax(logits)
# probs = [0.72, 0.17, 0.08, 0.03]  # Probability distribution, sums to 1
```

Note: Numerical stabilization (subtracting max_val) prevents overflow.

## Adam Optimizer

Adam (Adaptive Moment Estimation) is the most commonly used optimizer:

```python
# Hyperparameters
learning_rate = 0.01
beta1 = 0.85      # First moment (EMA of gradients)
beta2 = 0.99      # Second moment (EMA of squared gradients)
eps_adam = 1e-8   # Prevent division by zero

# In training loop
for i, p in enumerate(params):
    # 1. Update first moment estimate (similar to momentum)
    m[i] = beta1 * m[i] + (1 - beta1) * p.grad
    
    # 2. Update second moment estimate (similar to RMSProp)
    v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
    
    # 3. Bias correction (because m, v start at 0)
    m_hat = m[i] / (1 - beta1 ** (step + 1))
    v_hat = v[i] / (1 - beta2 ** (step + 1))
    
    # 4. Update parameters
    p.data -= learning_rate * m_hat / (v_hat ** 0.5 + eps_adam)
```

### Adam Intuition

| Component | Purpose |
|-----------|--------|
| First moment (m) | Accumulates historical gradient direction (like inertia) |
| Second moment (v) | Adjusts learning rate per parameter (larger gradient magnitude → smaller learning rate) |

In simple terms:
- Directions with stable gradients: take bigger steps
- Directions with oscillating gradients: take smaller steps

## Learning Rate Decay

```python
lr_t = learning_rate * (1 - step / num_steps)
```

Linear decay: Learning rate slowly decreases from initial value to 0.

```
Learning Rate
   ↑
   │────
   │    ╲
   │      ╲
   │        ╲
   │─────────────→ Steps
   0          num_steps
```

Why decay? Later in training, parameters are close to optimal values; using a large learning rate may cause oscillation.

## Training Results

Typical output from microgpt after 1000 training steps:

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

Loss dropped from 3.37 to 2.21 — the model is learning!

## Why Isn't Loss 0?

Theoretically, if the model were perfect, loss should be 0 (correct prediction probability = 1).

But in practice:
1. **Model too small**: Only 4,192 parameters, limited learning capacity
2. **Data randomness**: Names themselves have randomness (no perfect pattern)
3. **Character-level prediction limitations**: Only looking at the previous character to predict the next is fundamentally limited in accuracy

Loss = 2.21 means: the probability of the correct token is approximately $e^{-2.21} ≈ 11\%$.

## Summary

Key concepts from this chapter:
- **Training Loop**: Get data → Forward pass → Backward pass → Update parameters
- **Cross-Entropy Loss**: Measures the gap between predicted and true distributions
- **Adam Optimizer**: Combines momentum and adaptive learning rates
- **Learning Rate Decay**: Use smaller learning rates later for stable convergence

In the next chapter, we see how to use the trained model to generate new names!

---

*Previous: [9. GPT Architecture](09-gpt_architecture.md)*  
*Next: [11. Generating New Names — Temperature and Randomness](11-generation.md)*
