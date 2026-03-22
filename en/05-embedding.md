# 5. The Mathematics of Text—Tokenization and Embedding

## How Does a Computer Understand Text?

In previous chapters, we worked with numbers. But AI needs to process text—images, audio, and code can all be converted to text descriptions.

The question arises: **How does a computer "understand" text?**

The answer: **Convert text to numbers.**

## Tokenization: Text → Number Sequences

Tokenization is the process of splitting text into small chunks, assigning each chunk a numeric ID.

### A Simple Vocabulary

```python
vocab = {
    "我": 0,
    "愛": 1,
    "你": 2,
    "學": 3,
    "習": 4
}

text = "我愛學習"
tokens = [vocab[c] for c in text]
print(tokens)  # [0, 1, 3, 4]
```

### Character-level vs Word-level

| Method | Advantages | Disadvantages |
|--------|------------|---------------|
| Character-level | Small vocabulary, flexible with new characters | Longer token sequences |
| Word-level | Shorter sequences, clearer semantics | Large vocabulary, hard to handle new words |

### BPE: A Balance Between Both

BPE (Byte Pair Encoding) is a commonly used method in modern models:
- Treats common character combinations as single tokens
- Balances vocabulary size and sequence length
- Can handle unseen words (by splitting into known subwords)

tiktoken, used by GPT-4, is an implementation of BPE.

### microgpt's Tokenizer

microgpt uses the simplest **character-level Tokenizer**:

```python
# Build vocabulary from name list
docs = ["alex", "bob", "charlie", ...]
chars = sorted(set(''.join(docs)))  # All unique characters
vocab = {c: i for i, c in enumerate(chars)}

# vocab = {'a': 0, 'b': 1, 'c': 2, ..., 'z': 25, 'eos': 26}
```

## Embedding: Numbers → Vectors

The tokenizer gives each token an ID, but we need more information.

Question: For "a" with ID = 0 and "b" with ID = 1, what is the numerical relationship between them?

**It has no meaning!** Alphabetical order doesn't represent semantic similarity.

### Solution: Embedding

Map each token to a **vector** (a sequence of numbers):

```
"cat"  → [0.2, -0.5, 0.8, ...]  (300-dimensional vector)
"dog"  → [0.3, -0.4, 0.7, ...]  (similar!)
"car"  → [0.8, 0.2, -0.3, ...]  (unrelated)
```

### Where Do Embeddings Come From?

They start random and are learned during training:
1. Initialization: random vectors
2. Training: adjust vectors based on task
3. Result: words with similar semantics have similar vectors

## Vector Space: The Geometry of Meaning

Embedding vectorizes meaning; semantically similar things are close together in vector space.

### Analogy Reasoning

The most famous example:

```
King - man + woman ≈ Queen
```

Vector operations:

```python
king = embedding["king"]
man = embedding["man"]
woman = embedding["woman"]

queen ≈ king - man + woman
```

### Similarity: Dot Product

To measure how similar two vectors are, use **Dot Product**:

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# "cat" and "dog" are similar
similarity("cat", "dog") = 0.87

# "cat" and "car" are not very similar
similarity("cat", "car") = 0.23
```

## Positional Encoding: Position Information

Text has order, but neural networks don't initially know about positions.

**Positional Encoding** converts positions to vectors as well:

```
Position 0 token → [1, 0, 0, ...]
Position 1 token → [0, 1, 0, ...]
Position 2 token → [0, 0, 1, ...]
```

Transformers use sine/cosine functions to generate positional encoding:

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$$

## Transformer's Input Processing

Complete Transformer input pipeline:

```
Text → Token IDs → Embedding vectors → Add positional encoding → Input to Transformer
```

```python
# Illustrative code
token_ids = tokenizer("Hello world")  # [0, 1]
token_embeddings = embedding(token_ids)  # [[0.2, ...], [0.5, ...]]
position_embeddings = positional_encoding(2)
input_vectors = token_embeddings + position_embeddings  # Add position information
```

## Visualization: A Map of Vocabulary

Using dimensionality reduction techniques (like t-SNE, PCA), high-dimensional embeddings can be projected to 2D:

```
                    Animals
                   /    \
               cat — dog   Vehicles
                 \        /    \
                  bird  car
                       /
                   Colors
                   /
                 red - blue - green
```

Similar words cluster together.

## Embedding Dimensions

| Task | Dimensions | Description |
|------|------------|-------------|
| Small experiments | 16-64 | micrograd/microgpt |
| Small models | 128-512 | GPT-2 small |
| Medium models | 768-1024 | GPT-2 medium/large |
| Large models | 4096+ | GPT-4, etc. |

Higher dimensions can express finer semantics, but computational cost is also higher.

## Summary

Core concepts from this chapter:
- **Tokenization**: split text into chunks and assign numeric IDs
- **Embedding**: convert token IDs into meaningful vectors
- **Positional encoding**: add order information
- **Vector space**: semantically similar things are close in space

In the next chapter, we dive into the core of Transformers—the Attention Mechanism!

---

*Previous: [4. Understanding Automatic Differentiation with micrograd](04-micrograd.md)*  
*Next: [6. Transformer—The Art of Attention](06-transformer.md)*
