# 15. Prompt Engineering: The Art of Communicating with Models

## Why Are Prompts Important?

The same model can produce drastically different results with different prompts:

```python
# Prompt 1: Poor results
回答：「量子計算」
```

```python
# Prompt 2: Good results
"""
Explain the key points of quantum computing:
1. Use language that ordinary people can understand
2. Provide concrete examples
3. Do not exceed 200 words

Question: What is quantum computing?
"""
回答："Quantum computing is like flipping many coins simultaneously..."
```

Prompt Engineering is **learning how to communicate effectively with LLMs**.

## Basic Principles

### 1. Be Clear and Specific

| ❌ Vague | ✅ Specific |
|---------|------------|
| "Write a poem" | "Write a five-character quatrain about the first spring rain" |
| "Help me write code" | "Write a Python function to calculate Fibonacci sequence with type hints" |
| "Summarize this article" | "Summarize this article in three sentences, focusing on the author's arguments" |

### 2. Provide Context

```python
# No context
"Translate: Hello"

# With context
"""
You are a professional Chinese-English translator specializing in tech articles.
Please translate the following English text to Chinese, maintaining terminology consistency.

Original: The transformer architecture revolutionized NLP.
Translation:
"""
```

### 3. Specify Output Format

```python
# No format requirements
"List 5 ways to lose weight"

# With format requirements
"""
List 5 ways to lose weight in the following format:
{
  "method_1": "Method name",
  "description": "Detailed explanation"
}
"""
```

## Common Techniques

### Few-Shot Prompting

Provide a few examples for the model to learn patterns:

```python
prompt = """
Translate the following Chinese to English:

Example 1:
Chinese: 「今天天氣很好」
English: The weather is nice today.

Example 2:
Chinese: 「我愛學習新技能」
English: I love learning new skills.

Now translate:
Chinese: 「這本書非常有趣」
English:
"""
```

### Chain-of-Thought (CoT)

Encourage the model to think step by step:

```python
prompt = """
Question: Xiao Ming has 10 apples, gave Xiao Hua 3, and bought 5 more. How many apples does he have now?

Let's think step by step:
1. Xiao Ming started with 10 apples
2. He gave Xiao Hua 3, so he has 10 - 3 = 7 left
3. He bought 5 more, so he has 7 + 5 = 12

Answer: 12

Question: A car travels at 60 km/h for 2.5 hours. How far did it travel?
Let's think step by step:
"""
```

### Zero-Shot CoT

Even without examples, adding "Let's think step by step" improves results:

```python
prompt = """
Question: Three people stayed at a hotel, each paid 10 yuan, totaling 30 yuan. The owner said it's only 25 yuan and returned 5 yuan.
The waiter stole 2 yuan and returned 1 yuan to each person. So each person actually paid 9 yuan, 3*9=27 yuan,
the waiter stole 2 yuan, 27+2=29 yuan. Question: Where is the remaining 1 yuan?

Let's think step by step, don't rush to give an answer:
"""
```

### Tree of Thoughts (ToT)

Explore multiple thinking paths:

```python
prompt = """
When solving this problem, please consider multiple possible directions:

Question: How to increase company revenue by 50%?

Direction 1: Increase Revenue
- Add product lines
- Raise prices
- Expand into new markets

Direction 2: Reduce Costs
- Lower production costs
- Optimize staffing
- Reduce waste

Direction 3: Innovation
- Develop new technologies
- Improve customer experience
- Build partnerships

Please elaborate on the implementation steps for each direction.
"""
```

## Avoiding Common Mistakes

### 1. Context Length Limits

```python
# ❌ Too long, exceeds model's context limit
prompt = "Here is the content of a 100-page contract..." + "Please summarize the key points"

# ✅ Clearly indicate the focus
prompt = """
Here is a summary of key contract terms (pages 1-3):
[Summary content]

Based on the above summary, answer: What are the main obligations in this contract?
"""
```

### 2. Vague Constraints

| ❌ Vague Constraints | ✅ Specific Constraints |
|---------------------|------------------------|
| "Don't make it too long" | "Keep the response under 100 words" |
| "Be more professional" | "Use formal business language, avoid colloquial expressions" |
| "Be quick" | (Cannot be controlled, the model has its own speed) |

### 3. Conflicting Instructions

```python
# ❌ Conflicting
prompt = "Explain this concept in detail" + "Explain it in one sentence"

# ✅ Coordinated
prompt = "Explain the core concept in one sentence, with a brief example if needed"
```

## Advanced Techniques

### System Prompt

Set the role and behavior for the AI:

```python
messages = [
    {"role": "system", "content": "You are a senior software engineer skilled in Python and system design. You will provide code examples in your responses."},
    {"role": "user", "content": "How do I design a highly concurrent system?"}
]
```

### Refinement Loop

Iteratively improve responses:

```python
# Round 1: Provide initial response
回答1 = model.generate("Question: ...")

# Refine based on feedback
回答2 = model.generate(f"""
Original response: {回答1}

Please improve the response based on the following feedback:
- Add more concrete examples
- Simplify technical terminology
""")
```

## Summary

Key concepts from this chapter:
- **Be Clear and Specific**: Tell the model exactly what you want
- **Provide Context**: Help the model understand the background
- **Few-Shot**: Guide output with examples
- **Chain-of-Thought**: Encourage step-by-step thinking
- **System Prompt**: Set roles and behavior

In the next chapter, we'll look at what AI is good at and what it's not!

---

*Previous: [14. RLHF](14-rlhf.md)*  
*Next: [16. AI's Strengths and Weaknesses](16-capabilities_limits.md)*
