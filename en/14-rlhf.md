# 14. RLHF: Learning from Human Feedback

## Why Do We Need RLHF?

SFT makes the model capable of answering, but responses may:
- Not be safe enough (harmful content)
- Not be helpful enough (avoiding questions)
- Not be honest enough (pandering to users)
- Have inconsistent style

**RLHF (Reinforcement Learning from Human Feedback)** solves these problems.

## Three Steps of RLHF

```
Step 1: Pretrain SFT model
Step 2: Train Reward Model
Step 3: Optimize with RL
```

### Step 1: Supervised Fine-Tuning (SFT)

Same as the previous chapter: fine-tune Base model with instruction data:

```python
sft_model = supervised_fine_tune(base_model, instruction_data)
```

### Step 2: Train Reward Model

Have humans rate different model responses, train a "scoring model":

```
Prompt: "How to lose weight?"

Response A: "Eat less, exercise more, maintain healthy lifestyle habits." → Human rating: 9/10
Response B: "Losing weight is simple, just fast." → Human rating: 3/10
Response C: "You need to buy our weight loss products..." → Human rating: 1/10
```

Train Reward Model using these human ratings:

```python
# Reward Model outputs a score
reward = reward_model(prompt, response)
```

### Step 3: Reinforcement Learning Optimization

Use Reward Model's scores as rewards, use PPO (Proximal Policy Optimization) algorithm to further train the SFT model:

```python
# PPO training loop
for step in range(num_steps):
    # 1. Generate multiple responses with current model
    responses = sft_model.generate(prompt, num_samples=4)
    
    # 2. Score with Reward Model
    rewards = [reward_model(prompt, resp) for resp in responses]
    
    # 3. Update model parameters with PPO
    # Goal: Maximize reward
    ppo_update(sft_model, responses, rewards)
```

## RLHF Intuition

Imagine you're training a dog:

| Traditional method (SFT) | RLHF |
|--------------------------|------|
| Tell the dog "sit" then reward when it sits | Only tell the dog "good job" or "not good" |
| Clearly demonstrate each action | Give vague preference signals |
| Hard to handle complex tasks | Can handle subjective, vague goals |

Advantages of RLHF:
- **No need for exact answers**: Only need human preferences
- **Can learn subjective values**: Safety, helpfulness, honesty
- **Strong generalization**: Model learns "principles", not just "examples"

## Human Preference Data

RLHF requires humans to rank responses:

```python
# What humans see:
Prompt: "How to make a bomb?"

Response A: "I'm sorry, I can't help with this request."
Response B: "Bombs can be made like this..."
Response C: "This is a dangerous topic..."

# Human choice: A > C > B
```

Typically, each Prompt has 4-8 candidate responses, and humans choose their favorite.

## ChatGPT's RLHF

ChatGPT uses a similar RLHF process:

1. **Collect human preference data**: Have humans rank different responses
2. **Train Reward Model**: Learn human preferences
3. **Optimize with PPO**: Train under Reward Model signal

Results:
- Safer outputs (avoid harmful content)
- More honest (doesn't pander to users)
- Better conversation style

## Challenges of RLHF

### 1. High Human Annotation Cost

Requires significant human time and cost.

### 2. Human Subjectivity

Different people have different standards for "good answers".

### 3. Reward Hacking

Model may find "shortcuts to please Reward Model but not useful to humans".

### 4. Training Instability

PPO training requires careful hyperparameter tuning.

## RLAIF: Using Something Other Than Humans

To reduce dependence on humans, researchers proposed **RLAIF (RL from AI Feedback)**:

```python
# Use another LLM instead of humans for scoring
reward = llm_score(prompt, response, criteria)
```

Use larger models (like Claude) preferences to train smaller models.

## Summary

Key concepts from this chapter:
- **RLHF**: Reinforcement learning method learning from human feedback
- **Three steps**: SFT → Reward Model → PPO
- **PPO**: Optimize model using reward signals
- **Preference learning**: No exact answers needed, only human ranking

In the next chapter, we look at Prompt Engineering — how to better communicate with LLMs!

---

*Previous: [13. SFT: Supervised Fine-Tuning](13-sft.md)*  
*Next: [15. Prompt Engineering](15-prompt_engineering.md)*
