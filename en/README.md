# Let AI Teach You Modern AI

> Understanding gradient with micrograd, understanding LLMs with microgpt

---

## Table of Contents

### Part 1: Machine Learning Basics

* [1. The First Step of Machine Learning](01-ml_basics.md)
* [2. Learning from Mistakes - Gradient and Gradient Descent](02-gradient_descent.md)
* [3. Backpropagation - The Magic of Automatic Differentiation](03-backpropagation.md)
* [4. Understanding Automatic Differentiation with micrograd](04-micrograd.md)

### Part 2: Text Mathematics and Transformer

* [5. Mathematics of Text - Tokenization and Embedding](05-embedding.md)
* [6. Transformer - The Art of Attention](06-transformer.md)
* [7. Self-Attention Mechanism Explained](07-self_attention.md)

### Part 3: Understanding GPT from Scratch with microgpt

* [8. Complete Analysis of microgpt.py - Data, Tokenizer, Architecture](08-microgpt_overview.md)
* [9. GPT Architecture: Embedding → Transformer Layers → LM Head](09-gpt_architecture.md)
* [10. Training Loop and Adam Optimizer](10-training_loop.md)
* [11. Generating New Names - Temperature and Randomness](11-generation.md)

### Part 4: From microgpt to ChatGPT

* [12. The Power of Pretraining](12-pretraining.md)
* [13. SFT: Supervised Fine-Tuning](13-sft.md)
* [14. RLHF: Learning from Human Feedback](14-rlhf.md)
* [15. Prompt Engineering: The Art of Communicating with Models](15-prompt_engineering.md)

### Part 5: AI Capabilities, Limitations, and Future

* [16. What AI Excels At and Its Flaws](16-capabilities_limits.md)
* [17. Hallucination, Bias, and Safety](17-hallucination_safety.md)
* [18. Context Window and Agent](18-context_agent.md)
* [19. The Future of AI - Current Bottlenecks and Development Directions](19-future.md)

---

## Core Code

### Reference

* [micrograd](https://github.com/karpathy/micrograd) - Automatic Differentiation Engine
* [microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) - Complete GPT Implementation

### Book Examples

* [_code/nn0py/nn0.py](../_code/nn0py/nn0.py) - Autograd Engine (Automatic Differentiation)
* [_code/nn0py/gpt0.py](../_code/nn0py/gpt0.py) - GPT Model Implementation
* [_code/nn0py/cnn0.py](../_code/nn0py/cnn0.py) - CNN Model Implementation
* [_code/nn0py/test_gpt0.py](../_code/nn0py/test_gpt0.py) - GPT Training and Inference Test
* [_code/gd/gd.py](../_code/gd/gd.py) - Gradient Descent Implementation (Numerical Differentiation)
* [_code/gd/gd_array.py](../_code/gd/gd_array.py) - Vectorized Gradient Descent

---

*Last updated: 2026-03-22*
