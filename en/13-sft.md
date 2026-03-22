# 13. SFT: Supervised Fine-Tuning

## From Base Model to Instruction Model

The pretrained Base model is powerful, but doesn't understand human instructions well:

```
Base model input: "Explain quantum entanglement"
Base model output: "Quantum entanglement is a phenomenon in quantum mechanics..." 
(may continue writing "Quantum entanglement is..." 
—it is just "completing" text, not "answering" questions)
```

**SFT (Supervised Fine-Tuning)** transforms the Base model into one that understands instructions.

## What Is SFT?

The core idea of SFT is simple:

**Collect high-quality question-answer pairs written by humans, and fine-tune the Base model with this data.**

```
Question: What is quantum entanglement?
Answer: Quantum entanglement is a special relationship between two or more particles...
```

Let the model learn:
- Given a question → Generate an answer
- Follow instructions → Output helpful responses

## SFT Data Format

SFT uses data called **Instruction-Tuning datasets**:

```json
{
  "instruction": "Explain what quantum entanglement is",
  "input": "",
  "output": "Quantum entanglement is a phenomenon in quantum mechanics..."
}
```

Or a more complex format:

```json
{
  "instruction": "Translate the following text into English",
  "input": "你好，世界！",
  "output": "Hello, World!"
}
```

## Instruction Dataset Scale

| Dataset | Scale | Purpose |
|---------|-------|---------|
| FLAN | 1M+ | Various tasks |
| Alpaca | 52K | General instructions |
| Vicuna | 70K | Conversation |
| WizardLM | 700K | Complex instructions |

## SFT Training Process

```python
# 1. Load pretrained Base model
model = load_base_model("gpt-3")

# 2. Prepare instruction data
dataset = load_instruction_dataset("alpaca_data.json")

# 3. Fine-tune with instruction data
for epoch in range(3):
    for item in dataset:
        # Forward pass: Compute model prediction
        # Goal: Make prediction close to human-written answer
        loss = compute_loss(model, item)
        
        # Backpropagation + Update parameters
        loss.backward()
        optimizer.step()
```

## Training Details

### Learning Rate
Much lower than pretraining, typically 1/10 to 1/5 of the pretraining learning rate.

```python
# Pretraining learning rate: 0.0001
# SFT learning rate: 0.00001 ~ 0.00002
```

### Number of Epochs
Usually few (1-5 epochs) to avoid overfitting.

### Data Quality
Quality matters more than quantity. High-quality 10,000 examples are more effective than low-quality 1,000,000.

## Effects of SFT

| Capability | Base Model | After SFT |
|------------|------------|-----------|
| Understanding instructions | Poor | Good |
| Response quality | Unstable | Stable |
| Format correctness | Poor | Good |
| Safety | Poor | Better |

## Why Isn't SFT Enough?

SFT is effective, but has limitations:

1. **Requires heavy human labeling**: High cost, time-consuming
2. **Hard to scale**: New tasks require new data
3. **Style issues**: Model may just "imitate" the writing style of training data
4. **Limited safety**: SFT cannot fully solve harmful outputs

## Self-Instruct: Reducing Human Labeling

To reduce human labeling costs, researchers proposed the **Self-Instruct** method:

1. Use a few human-written examples
2. Let the model generate more instruction data
3. Fine-tune with this data

```python
# Prompt model to generate more using few examples
seed_examples = [
    {"instruction": "What is...?", "output": "..."},
    {"instruction": "How to...?", "output": "..."},
]

# Let model generate new instructions
new_instructions = model.generate(
    prompt=f"Based on the following examples, generate more similar Q&A:\n{seed_examples}"
)
```

Alpaca was created using the Self-Instruct method, generating data from GPT-3.5.

## Summary

Key concepts from this chapter:
- **SFT purpose**: Transform Base model into instruction-following model
- **Training data**: High-quality Q&A pairs written by humans
- **Training method**: Supervised learning, predict answers
- **Self-Instruct**: Generate training data using the model

In the next chapter, we look at RLHF — how models learn from human feedback to become safer and more helpful!

---

*Previous: [12. The Power of Pretraining](12-pretraining.md)*  
*Next: [14. RLHF: Learning from Human Feedback](14-rlhf.md)*
