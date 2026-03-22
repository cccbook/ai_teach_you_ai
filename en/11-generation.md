# 11. Generating New Names — Temperature and Randomness

## From Model to Generator

After training, we have a model that has "learned the patterns of names".

Now the question: **How do we make it generate new names?**

## Greedy Decoding: Always Pick the Highest Score

The simplest method: always pick the token with highest probability:

```python
def greedy_decode(logits):
    return np.argmax(logits)  # Pick the highest
```

Pros: Deterministic output
Cons: Lacks diversity, always the same

## Temperature: Controlling Randomness

microgpt uses **Temperature Sampling**:

```python
temperature = 0.5  # Between 0 and 1, smaller is more conservative

# Divide logits by temperature, then softmax
probs = softmax([l / temperature for l in logits])
token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
```

### Temperature Intuition

| Temperature | Effect | Analogy |
|-------------|--------|---------|
| High (e.g., 1.0) | High randomness, strong diversity | Coin flipped randomly |
| Low (e.g., 0.5) | Low randomness, more conservative | Coin slightly unfair |
| Close to 0 | Almost always picks the highest | Coin always heads |

### Mathematical Explanation

Original Softmax:

$$p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

With temperature:

$$p_i = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}$$

- When $T = 1$: Standard Softmax
- When $T \to 0$: Only the largest logit's probability approaches 1 (greedy)
- When $T \to \infty$: All logits approach uniform distribution

## microgpt's Generation Code

```python
temperature = 0.5

print("--- inference (new, hallucinated names) ---")
for sample_idx in range(20):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS  # Start from BOS
    sample = []
    
    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)
        
        # Softmax with temperature
        probs = softmax([l / temperature for l in logits])
        
        # Sample based on probabilities
        token_id = random.choices(range(vocab_size), 
                                   weights=[p.data for p in probs])[0]
        
        if token_id == BOS:  # Stop when encountering BOS
            break
        
        sample.append(uchars[token_id])
    
    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")
```

## Generation Flow Diagram

```
BOS ──→ Model ──→ logits ──→ softmax(temperature) ──→ Sample ──→ 'a'
                                                                  │
                                                                  ↓
BOS, a ──→ Model ──→ logits ──→ softmax(temperature) ──→ Sample ──→ 'l'
                                                                            │
                                                                            ↓
BOS, a, l ──→ Model ──→ logits ──→ softmax(temperature) ──→ Sample ──→ 'e'
                                                                                   │
                                                                                   ↓
BOS, a, l, e ──→ Model ──→ ... ──→ Sample ──→ 'x'
                                                     │
                                                     ↓
BOS, a, l, e, x ──→ Model ──→ logits ──→ BOS ──→ Stop!
                                                     │
                                                     ↓
                                             Output: "alex"
```

## Generation Results

Results generated with temperature 0.5 after 1000 training steps:

```
sample  1: ari
sample  2: karia
sample  3: alex
sample  4: michon
sample  5: ryan
sample  6: emily
sample  7: davi
sample  8: sophia
sample  9: andrew
sample 10: lexa
```

These names look like real names! The model learned:
- Common letter combinations (alex, emily, andrew)
- Name structure (multiple samples look like real English names)

## Effects of Different Temperatures

| Temperature | Output Characteristics | Examples |
|------------|----------------------|----------|
| 0.2 | Very conservative, common combinations | alex, emily, andrew |
| 0.5 | Balanced diversity | ari, karia, michon |
| 1.0 | High diversity, some strange combinations | axzh, kreoql, vnm |
| 2.0 | Almost random | May not look like names at all |

## Why Is It Called "Hallucinated"?

Karpathy used the word "hallucinated" in the code, meaning "imagined" or "made up".

These names **don't exist in the training data**, but look plausible — like the AI is "hallucinating" new names.

The same concept applies: ChatGPT sometimes produces "hallucinations" — outputs that look reasonable but are actually incorrect.

## Autoregressive Generation: Step by Step

This generation method is called **Autoregressive**:

```
t=0: Input BOS → Predict w₁
t=1: Input BOS, w₁ → Predict w₂
t=2: Input BOS, w₁, w₂ → Predict w₃
...
```

Each step's output becomes the next step's input.

Pros: Can generate sequences of arbitrary length
Cons: Slow generation (cannot be parallelized)

## Comparison with Modern LLMs

| Feature | microgpt | GPT-4 |
|---------|---------|-------|
| Generation method | Autoregressive | Autoregressive |
| Temperature control | Yes | Yes |
| Additional techniques | None | Beam Search, Top-k, Top-p |

### Top-k Sampling

```python
# Sample only from the top k tokens
top_k = 50
indices = np.argsort(logits)[-top_k:]  # Indices of top 50 largest
probs_topk = softmax([logits[i] for i in indices])
token_id = random.choices(indices, weights=probs_topk)[0]
```

### Top-p (Nucleus) Sampling

```python
# Sample from the smallest set whose cumulative probability exceeds p
sorted_probs = sorted(probs, reverse=True)
cumsum = np.cumsum(sorted_probs)
mask = cumsum < 0.9  # Accumulate up to 90%
token_id = random.choices(np.where(mask)[0])[0]
```

## Summary

Key concepts from this chapter:
- **Greedy decoding**: Always pick the highest-scoring token
- **Temperature sampling**: Control output randomness
- **Autoregressive generation**: Generate sequences step by step
- **Hallucination**: Generating content that looks plausible but doesn't actually exist/is incorrect

In the next chapter, we upgrade from microgpt to real LLMs — understanding the power of pretraining!

---

*Previous: [10. Training Loop](10-training_loop.md)*  
*Next: [12. The Power of Pretraining](12-pretraining.md)*
