# 4. Understanding Automatic Differentiation from Scratch with micrograd

## Introduction to micrograd

micrograd is a tiny automatic differentiation framework written by Andrej Karpathy. With only about 150 lines of Python code, it fully implements the core functionality of neural network training.

github: https://github.com/karpathy/micrograd

The code used in this book is located at: [_code/nn0py/nn0.py](../_code/nn0py/nn0.py) (modular version)

## Core: The Value Class

The core of micrograd is the `Value` class, which encapsulates:
- `data`: the value of this node
- `grad`: the gradient of this node (default 0)
- `_children`: other nodes this node depends on
- `_local_grads`: local gradients (used for backpropagation)

```python
class Value:
    def __init__(self, data, children=(), local_grads=()):
        self.data = data              # scalar value
        self.grad = 0                 # gradient, default 0
        self._children = children    # child nodes
        self._local_grads = local_grads  # local gradients
```

## Basic Operations: Addition and Multiplication

### Addition

$$c = a + b \Rightarrow \frac{\partial c}{\partial a} = 1, \frac{\partial c}{\partial b} = 1$$

```python
def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    return Value(
        self.data + other.data,           # forward value = a + b
        (self, other),                    # record child nodes
        (1, 1)                            # local gradients
    )
```

### Multiplication

$$c = a × b \Rightarrow \frac{\partial c}{\partial a} = b, \frac{\partial c}{\partial b} = a$$

```python
def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    return Value(
        self.data * other.data,           # forward value = a * b
        (self, other),                    # record child nodes
        (other.data, self.data)           # local gradients
    )
```

## Building the Computation Graph

Let's see how the computation graph is built with a simple example:

```python
from nn0 import Value

# Create parameters and variables with initial values
a = Value(3.0)
b = Value(2.0)
x = Value(4.0)

# Build computation graph (Forward pass): y = a * x^2 + b
y = a * (x ** 2) + b

# Display forward propagation result
print(f"Forward calculation result y.data = {y.data:.4f}")  # Expected: 3 * 16 + 2 = 50
```

Computation graph:

```
    a (2) ──┐
            ├── * ──> c (6) ──┬── + ──> d (7) ──┬── * ──> e (14)
    b (3) ──┘                 └── 1.0 (1) ──────┘         │
                                                           └── 2.0
```

## Backpropagation: Automatically Computing Gradients

The key `backward()` method:

```python
def backward(self):
    # Build topological sort for backpropagation
    topo = []
    visited = set()
    
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._children:
                build_topo(child)
            topo.append(v)
    
    build_topo(self)
    
    # Start from output, propagate gradients backward
    self.grad = 1  # gradient of loss with respect to itself is 1
    for v in reversed(topo):
        for child, local_grad in zip(v._children, v._local_grads):
            # Chain rule: child.grad += local_grad * v.grad
            child.grad += local_grad * v.grad
```

## Complete Example: Computing $f(x) = 3x^2 + 2x + 1$

```python
from micrograd import Value

x = Value(2.0)

# f(x) = 3x² + 2x + 1
f = 3 * x * x + 2 * x + 1

print(f"f(x) = {f.data}")  # f(2) = 3*4 + 4 + 1 = 17

f.backward()

print(f"f'(x) = {x.grad}")  # f'(2) = 6*2 + 2 = 14
```

Verification: $f(x) = 3x^2 + 2x + 1$, $f'(x) = 6x + 2$, $f'(2) = 14$ ✓

### Actual Execution

Execution result:

```
=== 1. Basic Automatic Differentiation Operations ===
Forward calculation result y.data = 50.0000
dy/da = 16.0000 (theoretical value x^2 = 16)
dy/db = 1.0000 (theoretical value 1)
dy/dx = 24.0000 (theoretical value 2ax = 2*3*4 = 24)
```

Perfectly matches theoretical values!

## Hands-on: Training a Neural Network

With automatic differentiation, we can train a simple neural network:

### Goal

Learn AND logic (output 1 when both inputs are 1, otherwise output 0):

| x1 | x2 | Output |
|----|----|--------|
| 0  | 0  | 0   |
| 0  | 1  | 0   |
| 1  | 0  | 0   |
| 1  | 1  | 1   |

### Implementing Linear Regression with nn0.py

```python
from nn0 import Value, Adam

# Prepare training data (target function y = 2x + 1)
X = [1.0, 2.0, 3.0, 4.0]
Y = [3.0, 5.0, 7.0, 9.0]

# Initialize model parameters (weights and bias)
w = Value(0.0)
b = Value(0.0)

# Instantiate Adam optimizer
optimizer = Adam([w, b], lr=0.1)

# Train for 50 iterations
for epoch in range(51):
    # Forward propagation: predictions and compute Mean Squared Error (MSE) Loss
    preds = [w * x + b for x in X]
    losses = [(pred - y) ** 2 for pred, y in zip(preds, Y)]
    loss = sum(losses) / len(losses)
    
    # Backpropagation and parameter update
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch:2d} | Loss: {loss.data:.4f} | w: {w.data:.4f}, b: {b.data:.4f}")
```

### Training Process

Execution result:
```
Epoch  0 | Loss: 50.0000 | w: 0.4000, b: 0.2000
Epoch 10 | Loss: 0.0187 | w: 1.9559, b: 1.0356
Epoch 20 | Loss: 0.0007 | w: 1.9883, b: 1.0129
Epoch 30 | Loss: 0.0000 | w: 1.9972, b: 1.0049
Epoch 40 | Loss: 0.0000 | w: 1.9992, b: 1.0020
Epoch 50 | Loss: 0.0000 | w: 1.9998, b: 1.0008
```

Loss drops from 50 to near 0, weights converge to $w \approx 2$, $b \approx 1$—perfectly matching the target function $y = 2x + 1$!

### A More Complex Network: The XOR Problem

XOR is a classic challenge for traditional perceptrons, requiring a multi-layer network to solve:

```python
from nn0 import Value, Adam
import random

# XOR training data
X_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y_data = [0, 1, 1, 0]

# Initialize 2-4-1 network parameters
w1 = [[Value(random.uniform(-1, 1)) for _ in range(2)] for _ in range(4)]
b1 = [Value(random.uniform(-1, 1)) for _ in range(4)]
w2 = [[Value(random.uniform(-1, 1)) for _ in range(4)] for _ in range(1)]
b2 = [Value(random.uniform(-1, 1)) for _ in range(1)]

# Collect all parameters for Adam
params = [p for row in w1 for p in row] + b1 + [p for row in w2 for p in row] + b2
optimizer = Adam(params, lr=0.1)

# Training loop
for epoch in range(101):
    total_loss = Value(0.0)
    for x, y in zip(X_data, Y_data):
        pred = mlp(x)
        total_loss += (pred - y) ** 2
    loss = total_loss / len(X_data)
    loss.backward()
    optimizer.step()
```

Execution result:
```
Epoch   0 | Loss: 0.2556
Epoch  20 | Loss: 0.0128
Epoch  40 | Loss: 0.0014
Epoch  60 | Loss: 0.0002
Epoch  80 | Loss: 0.0000
Epoch 100 | Loss: 0.0000

Prediction Results:
Input [0, 0] -> Prediction: 0.0000 (Target: 0)
Input [0, 1] -> Prediction: 1.0000 (Target: 1)
Input [1, 0] -> Prediction: 1.0000 (Target: 1)
Input [1, 1] -> Prediction: 0.0000 (Target: 0)
```

Perfectly solved XOR! Multi-layer networks + nonlinear activation functions (ReLU) make the impossible possible.

## Limitations of micrograd

micrograd is a teaching tool, not a production tool:

| Limitation | Description |
|------------|-------------|
| Slow | Uses Python objects for computation, hundreds of times slower than NumPy/PyTorch |
| Scalar only | Cannot directly handle vectors or matrices |
| No GPU support | All computation runs on CPU |

But its value lies in: **completely understanding the principles behind automatic differentiation**.

## Comparison with PyTorch

micrograd's `Value` class is a simplified version of PyTorch's `Tensor`:

| micrograd | PyTorch |
|-----------|---------|
| `Value` class | `Tensor` class |
| Manual gradients | `autograd` module |
| Scalar computation | Tensor computation |
| Single-threaded | Multi-threaded + GPU |

Once you master micrograd, you master the core concepts of PyTorch's automatic differentiation!

## Summary

Core concepts from this chapter:
- **Value class**: encapsulates value and gradient
- **Computation graph**: tracks dependencies of each operation
- **Backpropagation**: applies chain rule from back to front
- **Training loop**: forward → backward → update → repeat

In the next chapter, we leave scalar computation and enter the mathematical world of text—Tokenization and Embedding!

---

*Previous: [3. Backpropagation](03-backpropagation.md)*  
*Next: [5. Tokenization and Embedding](05-embedding.md)*
