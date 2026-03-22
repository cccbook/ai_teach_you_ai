# 3. Backpropagation - The Magic of Automatic Differentiation

## Why Do We Need Backpropagation?

In the previous chapter, we learned gradient descent: updating parameters along negative gradient to minimize loss.

But here's the question: **How do we compute gradients for complex neural networks?**

A modern neural network may have:
- Hundreds of millions of parameters
- Hundreds of layers
- Complex non-linear operations

Manually computing gradients for each parameter is impossible. This is where **Backpropagation** comes in.

## Core Idea: Chain Rule

The mathematical foundation of backpropagation is the **Chain Rule**.

### What is the Chain Rule?

If y is a function of u, and u is a function of x:

$$y = f(u) \quad u = g(x)$$

Then the derivative of y with respect to x is:

$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$

Simple example:

$$y = \sin(x^2)$$

Let $u = x^2$, then $y = \sin(u)$

$$\frac{dy}{dx} = \cos(u) \cdot 2x = 2x \cos(x^2)$$

### Why is the Chain Rule So Important?

Because a neural network is a long chain of functions:

$$loss = f_4(f_3(f_2(f_1(x))))$$

Applying the chain rule:

$$\frac{\partial loss}{\partial x} = \frac{\partial f_4}{\partial f_3} \cdot \frac{\partial f_3}{\partial f_2} \cdot \frac{\partial f_2}{\partial f_1} \cdot \frac{\partial f_1}{\partial x}$$

From output back to input, layer by layer "passing back" - this is the origin of "backward".

## Forward Pass vs Backward Pass

```
Forward Pass:
Input → Layer 1 → Layer 2 → Layer 3 → Output → Calculate Loss

Backward Pass:
Loss ← Layer 3 Gradient ← Layer 2 Gradient ← Layer 1 Gradient ← Input Gradient
```

### Forward Pass

Data flows from input to output, each layer produces intermediate results.

```python
# Simplified neural network
h1 = x * w1 + b1      # First layer
h2 = relu(h1)          # Activation function
out = h2 * w2 + b2     # Output layer
loss = (out - y)**2    # Calculate loss
```

### Backward Pass

Starting from the loss function, compute gradients layer by layer using the chain rule.

```python
# Backward pass
d_out = 2 * (out - y)       # ∂loss/∂out
d_w2 = d_out * h2            # ∂loss/∂w2
d_b2 = d_out                 # ∂loss/∂b2
d_h2 = d_out * w2           # ∂loss/∂h2
d_relu = d_h2 * (h1 > 0)   # ∂relu/∂h1
d_w1 = d_relu * x           # ∂loss/∂w1
d_b1 = d_relu               # ∂loss/∂b1
```

## Computation Graph: Visual Understanding

Let's visualize with a computation graph:

```
       x ─┬─ w1 ─┬─ relu ─┬─ w2 ─┬─ out ─┬─ loss
            │       │         │       │        │
           2        3         4       5        6
```

### Forward Pass (Red arrows):
```
x=1 → multiply by w1=2 → get 2 → relu(2)=2 → multiply by w2=3 → get 6 → calculate loss
```

### Backward Pass (Blue arrows):
```
∂loss/∂out=1 → multiply by w2=3 → get ∂loss/∂relu=3 → relu derivative=1 → get ∂loss/∂(wx+b)=3 → multiply by x=1 → get ∂loss/∂w1=3
```

## Automatic Differentiation: Let Computers Compute Gradients Automatically

Manually computing gradients is painful and error-prone.

**Automatic Differentiation** automates this process.

Principles:
1. Decompose operations into basic operations (add, subtract, multiply, divide, relu, log, exp...)
2. Each basic operation defines its own gradient
3. Record computation graph during forward pass
4. Automatically apply chain rule during backward pass

This is the core principle behind `micrograd` and frameworks like PyTorch.

## Key Operations in Backpropagation

### Matrix Multiplication

```python
# Forward: y = W @ x
y = W @ x

# Backward: compute using upstream gradients and W^T
d_x = W.T @ d_y
d_W = d_y @ x.T
```

### ReLU Activation Function

```python
def relu(x):
    return max(0, x)

# Forward: y = relu(x)
y = max(0, x)

# Backward:
# If x > 0: dy/dx = 1, so upstream gradient passes through directly
# If x <= 0: dy/dx = 0, so gradient is 0
d_x = d_y * (x > 0)
```

### Softmax Function

```python
# Forward
exp_x = exp(x - max(x))  # Numerical stabilization
y = exp_x / sum(exp_x)

# Backward (more complex)
# We'll cover this in detail in Chapter 6 microgpt
```

## Why Are GPUs Good at This?

Neural network training requires massive matrix operations - exactly what GPUs excel at:

| Feature | CPU | GPU |
|---------|-----|-----|
| Core Count | A few to dozens | Thousands |
| Design Target | Complex logic | Massive parallel operations |
| Matrix Multiplication Speed | Slow | 100-1000x faster |

Neural network gradient computation is highly parallel, perfectly suited for GPU strengths.

## Summary

Key concepts from this chapter:
- **Chain Rule**: Core mathematical tool for computing derivatives of composite functions
- **Backpropagation**: Compute gradients from back to front along computation graph
- **Forward Pass**: Data flows from input to output
- **Automatic Differentiation**: Computers automatically compute gradients for complex functions

In the next chapter, we'll get hands-on with micrograd, a tiny framework, to deeply understand automatic differentiation!

---

*Previous: [2. Gradient and Gradient Descent](02-gradient_descent.md)*  
*Next: [4. Understanding Automatic Differentiation with micrograd](04-micrograd.md)*
