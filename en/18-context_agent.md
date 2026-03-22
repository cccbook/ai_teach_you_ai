# 18. Context Windows and Agents

## Context Window: AI's "Memory"

**Context Window** is the maximum text length an LLM can process.

### Why Are There Limits?

Transformer's attention mechanism needs to calculate relationships between each token and all other tokens:

$$O(n^2 \cdot d)$$

Where $n$ is the sequence length and $d$ is the dimension. As $n$ grows, computation increases quadratically.

| Model | Context Length | Approximate Equivalent |
|-------|---------------|----------------------|
| GPT-3.5 | 4K - 16K tokens | ~3000 - 12000 Chinese characters |
| GPT-4 | 8K - 128K tokens | ~6000 - 96000 Chinese characters |
| Claude 2 | 200K tokens | ~150000 Chinese characters |

### Impact of Context Window

```python
# ✅ Within context window
prompt = """
Here is the complete customer complaint record (5000 characters):
[5000 characters of content]
Please summarize the key points of this complaint.
"""

# ❌ Exceeds context window
prompt = """
Here are all customer emails from the past ten years (1 million characters):
[1 million characters of content]
Please analyze customer satisfaction trends.
"""
```

## How to Handle Limited Context?

### 1. Summarization and Compression

```python
# Original: 100-page document
# Step 1: Have AI summarize each chapter
# Step 2: Answer questions based on summaries
```

### 2. RAG (Retrieval-Augmented Generation)

```
User question → Retrieve relevant documents → Add to context → AI answers
```

```python
# 1. Split long document into chunks
chunks = split_document("long_document.txt", chunk_size=500)

# 2. Build vector index
index = build_vector_index(chunks)

# 3. Retrieve relevant content
relevant_chunks = search(index, "user question")

# 4. Combine into prompt
prompt = f"""
Relevant document content:
{relevant_chunks}

Question: {user_question}
"""
```

### 3. Chunked Processing

```python
# Split long task into multiple short tasks
task = "Analyze all characters in this book"
chapters = ["Chapter 1", "Chapter 2", ...]

for chapter in chapters:
    analysis = ai.analyze(chapter, "Analyze the characters in this chapter")
    all_analyses.append(analysis)

final_summary = ai.summarize(all_analyses, "Summarize all character analyses")
```

## Agents: Enabling AI to Take Action

**AI Agent** is an AI system that can autonomously plan, use tools, and execute actions.

### Core Agent Capabilities

```
┌─────────────────────────────────────┐
│           AI Agent                   │
├─────────────────────────────────────┤
│  Perception                          │
│    Understanding user input, environmental info
├─────────────────────────────────────┤
│  Planning                            │
│    Decomposing tasks, formulating action plans
├─────────────────────────────────────┤
│  Action                              │
│    Using tools, calling APIs, executing operations
├─────────────────────────────────────┤
│  Memory                              │
│    Storing conversation history, learning experiences
└─────────────────────────────────────┘
```

### Tool Use: Enabling AI to Use Tools

LLMs can only generate text, but through **Function Calling**, they can use external tools:

```python
# Define tools
tools = [
    {
        "name": "search",
        "description": "Search the web for the latest information",
        "parameters": {"query": "string"}
    },
    {
        "name": "calculate",
        "description": "Perform mathematical calculations",
        "parameters": {"expression": "string"}
    },
    {
        "name": "send_email",
        "description": "Send an email",
        "parameters": {"to": "string", "content": "string"}
    }
]

# AI can call these tools
user_input = "Search for the latest AI news, then send a summary to my teacher via email"

# AI's plan:
# 1. Call search({"query": "latest AI news"})
# 2. Organize summary
# 3. Call send_email({"to": "teacher", "content": "summary"})
```

### ReAct: Think + Act + Observe

Enabling AI to combine actions during the thinking process:

```python
prompt = """
You are a travel assistant. For each question:
1. Thought: Think about what to do
2. Action: Execute an action (use a tool)
3. Observation: Observe the result
4. Continue thinking or give the final answer based on observation

User question: I'm going to Taipei this weekend and want to know the weather, and recommend a nearby attraction.

Thought: I need to check the weather forecast for Taipei first
Action: call_weather(city="Taipei")
Observation: Weather forecast shows clear skies on the weekend, temperature 25-30 degrees
Thought: The weather is great, I can recommend outdoor attractions
Action: call_search(query="Taipei weekend outdoor attractions recommendation")
Observation: Found the following attractions: Yangmingshan National Park, Tamsui Old Street, Shilin Night Market...
Final: Based on the weather and search results, I recommend going to Yangmingshan National Park this weekend...
"""
```

### AI Agents in Self-Driving Cars

```python
# Perception
perception = {
    "cameras": "Car ahead detected",  # Image recognition
    "lidar": "30 meters from car ahead",  # Distance measurement
    "maps": "On Zhongxiao East Road"  # Localization
}

# Planning
plan = {
    "goal": "Stay in lane center, turn right after 100 meters",
    "speed": "Maintain 40 km/h",
    "emergency_plan": "Brake immediately if someone enters"
}

# Action
actions = {
    "steering": "Maintain straight angle",
    "throttle": "Maintain current throttle",
    "brake": "Standby"
}
```

## Agent Application Scenarios

| Application | Description |
|-------------|-------------|
| Personal assistant | Help with emails, schedules, tasks |
| Programming assistant | Auto write code, debug, refactor |
| Research assistant | Search literature, organize data, write reports |
| Process automation | Auto fill forms, process documents, approval workflows |

## Agent Challenges

### 1. Planning Failures

Complex task planning can go wrong:

```python
# Problem: Assembling furniture
# Actually: Wrong part order leads to failed assembly
```

### 2. Tool Use Errors

LLM may call the wrong tool or use wrong parameters:

```python
# Intent: Search for weather
# Actually: Called wrong API, returned incorrect results
```

### 3. Error Accumulation

Small errors at each step can accumulate into big problems:

```
Step 1: Small error →
Step 2: Based on wrong input →
Step 3: Bigger error →
... →
Final: Completely wrong result
```

### 4. Security Risks

Agents that can execute actions may cause greater harm:

```python
# Problem: AI autonomously decides to send mass emails
# Result: Labeled as spam sender, causing losses
```

## Summary

Key concepts from this chapter:
- **Context Window**: LLM's "memory" limitation
- **Solutions**: Summarization, RAG, chunked processing
- **Agent**: AI system that can perceive, plan, and act
- **Tool Use**: Enabling AI to use external tools
- **Challenges**: Planning failures, error accumulation, security risks

In the next chapter, we'll look at the future of AI!

---

*Previous: [17. Hallucination, Bias, and Safety](17-hallucination_safety.md)*  
*Next: [19. The Future of AI](19-future.md)*
