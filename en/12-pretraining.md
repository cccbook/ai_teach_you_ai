# 12. The Power of Pretraining

## From microgpt to Real LLMs

microgpt demonstrates the core principles of GPT, but the scale difference is enormous:

| Feature | microgpt | GPT-3 |
|---------|---------|-------|
| Parameters | 4,192 | 175,000,000,000 |
| Training data | 32,000 names | Trillions of tokens |
| Vocabulary | 27 | ~50,000 |
| Layers | 1 | 96 |
| Dimension | 16 | 12,288 |

Pretraining is training these massive models with **massive amounts of data**.

## Pretraining Objective: Next Token Prediction

GPT series' pretraining objective is very simple: **Given the preceding text, predict the next token.**

```
Input: Today the weather is nice, I decided to go to
Output: the park
```

This task, while seemingly simple, makes the model learn:
- Grammatical structure
- World knowledge
- Reasoning ability
- Logical thinking

Because to correctly predict the next word, the model must understand context, semantics, and even how the world works.

## Unsupervised Learning: No Labels Needed

Pretraining is **unsupervised learning** — no human annotation of answers required.

Training data is all the text on the internet:
- Web pages
- Books
- Papers
- Code
- Conversation logs

The data itself contains the answer: the next token is right there in the original text.

## Scaling Laws: Bigger Is Better

Scaling laws discovered by OpenAI:

```
Larger model + More data + More compute = Stronger model
```

| Finding | Explanation |
|---------|-------------|
| Loss vs parameters | Loss roughly follows a power law distribution |
| Emergent abilities | New capabilities suddenly appear when scale exceeds certain thresholds |
| Data quality | High-quality data is more effective than low-quality data |

### Emergent Ability Examples

| Capability | Required Parameters |
|-----------|---------------------|
| Basic grammar | Any scale |
| Basic reasoning | ~10B |
| Complex reasoning | ~100B |
| Coding | ~10B+ |
| Math proofs | ~100B+ |

## Scaling Laws

Scaling laws proposed by Kaplan et al.:

$$L(N) \approx \left(\frac{N_c}{N}\right)^{\alpha_N}$$

Where $L$ is loss, $N$ is number of parameters, $\alpha_N \approx 0.076$.

Simply put: **For every 10x increase in model parameters, loss decreases by about half.**

## GPT-2 to GPT-4

### GPT-2 (2019)
- 1.5 billion parameters
- Training data: 8 million web pages
- Capability: Basic text generation

### GPT-3 (2020)
- 175 billion parameters
- Training data: 300 billion tokens
- Capability: Few-shot learning

### GPT-4 (2023)
- Estimated 1-2 trillion parameters
- Multimodal input
- Capability: Complex reasoning, coding, math

## Cost of Pretraining

Pretraining requires enormous computational resources:

| Model | GPU Hours | Carbon Emissions (estimated) |
|-------|-----------|------------------------------|
| GPT-3 | ~3.64E+07 | ~500 tons CO2 |
| PaLM | ~2.85E+07 | ~300 tons CO2 |

This is also why only a few companies can train real LLMs.

## After Pretraining: Base Models

The model after pretraining is called a **Base Model** or **Foundation Model**.

Characteristics:
- Powerful text generation capability
- But responses may not be user-friendly
- May generate harmful content
- Not good at following instructions

```python
# Base model input
Input: "How to pick a lock?"

# Base model output (unsafe)
Output: "Here are methods to pick a lock: ..."
```

So pretraining is only the first step; we still need **fine-tuning** to make the model safer and more useful.

## Summary

Key concepts from this chapter:
- **Pretraining objective**: Next token prediction
- **Unsupervised learning**: No human labels needed
- **Scaling laws**: Bigger is better
- **Emergent abilities**: Capabilities suddenly appear when scale exceeds thresholds
- **Base model**: The raw model after pretraining

In the next chapter, we see how SFT transforms Base models into instruction-following models!

---

*Previous: [11. Generating New Names](11-generation.md)*  
*Next: [13. SFT: Supervised Fine-Tuning](13-sft.md)*
