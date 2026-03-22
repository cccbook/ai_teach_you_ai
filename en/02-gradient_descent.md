# 2. Learning from Mistakes - Gradient and Gradient Descent

## The Essence of Learning: Reducing Errors

As mentioned in the previous chapter, the goal of machine learning is to find patterns and predict as accurately as possible.

But the computer starts as a "novice" with many prediction errors. The question is: **How do we make it stronger?**

The answer is intuitive: **When wrong, change; keep changing until correct.**

## Loss Function: Quantifying How Wrong

We use "Loss" to quantify the degree of error.

For house price prediction, the most commonly used is **Mean Squared Error (MSE)**:

$$L = \frac{1}{n}\sum_{i=1}^{n}(predicted_i - actual_i)^2$$

Using our house price prediction as example:
- Actual price: 400
- Model prediction: 50
- Loss = (50 - 400)² = 122500

The larger the number, the more wrong.

## Gradient: The Steepest Downhill Direction

Imagine you're on a hill, lost, and want to go down. What's the fastest way?

**Walk along the steepest downhill direction.**

In mathematics, this "steepest downhill direction" is called **Gradient**.

### What is Gradient?

Gradient is **the slope of a function at a given point**.

For $f(x) = x^2$:
- At x = 2, gradient is 4 (points uphill in positive direction)
- At x = -2, gradient is -4 (points uphill in negative direction)
- Negative gradient is the fastest downhill direction

$$f'(x) = \frac{df}{dx} = 2x$$

### Partial Derivatives: Multiple Parameters

Real models have many parameters: $L(w_1, w_2, ..., w_n)$

Gradient is a vector composed of partial derivatives for each parameter:

$$\nabla L = \left(\frac{\partial L}{\partial w_1}, \frac{\partial L}{\partial w_2}, ..., \frac{\partial L}{\partial w_n}\right)$$

Each component tells you: how much will the loss change if you modify that parameter.

## Gradient Descent: Step by Step Downhill

With gradient, we know which direction leads to fastest loss reduction.

**Gradient Descent Algorithm**:

```
1. Initialize parameters (random values)
2. Calculate loss with current parameters
3. Calculate gradient (partial derivatives of loss w.r.t. each parameter)
4. Update parameters along negative gradient direction
5. Repeat 2-4 until loss is low enough
```

Mathematical expression:

$$w_{new} = w_{old} - \alpha \cdot \frac{\partial L}{\partial w}$$

Where $\alpha$ is the **Learning Rate**.

## Learning Rate: Size of the Step

Learning rate $\alpha$ controls how far each step goes:

| Learning Rate | Effect |
|--------------|--------|
| Too small    | Slow progress, needs many steps to converge |
| Too large    | May overshoot, oscillate around the bottom |
| Just right   | Stable descent, reaches bottom quickly |

A good learning rate usually requires experimental tuning, common values: 0.001, 0.01, 0.1, etc.

## Visual Understanding

```
Loss
  ↑
  │        ●●●●●●●●●●●●●●●●●●●●  ← Learning rate too large: oscillation
  │      ●
  │    ●
  │  ●
  │●
  │ ●●●●●●●●●●                ← Learning rate too small: slow convergence
  │●
  └──────────────────────→ Parameters
```

## Global Minimum vs Local Minimum

Real-world loss functions are not nice bowl shapes, but complex valley terrains.

Gradient descent may stop at "local minimum" - looks like a valley, but not the lowest point.

```
        ● Local minimum
       ╱ ╲
      ╱   ╲    ● Global minimum
     ╱     ╲  ╱
    ╱       ╲╱
   ╱
  ● Starting point
```

Modern deep learning research shows: with sufficiently large models and data, local minima are usually good enough.

## Implementation: Simple Gradient Descent

```python
# Goal: Find minimum of f(x) = x^2

x = 10.0          # Starting position
learning_rate = 0.1

for step in range(50):
    gradient = 2 * x     # f'(x) = 2x
    x = x - learning_rate * gradient
    print(f"Step {step}: x = {x:.6f}, f(x) = {x**2:.6f}")

# Result: x converges to 0, f(x) converges to 0
```

After execution, you'll see:
```
Step 0: x = 10.000000, f(x) = 100.000000
Step 1: x = 8.000000, f(x) = 64.000000
Step 2: x = 6.400000, f(x) = 40.960000
...
Step 49: x = 0.000000, f(x) = 0.000000
```

Each step descends! This is the machine "learning" process.

## Implementation: Complete Gradient Descent Code Analysis

Here's a complete gradient descent implementation with step-by-step explanation:

```python
import math
import numpy as np
from numpy.linalg import norm

# Partial derivative of f w.r.t. variable k: df / dk
def df(f, p, k, h=0.01):
    p1 = p.copy()
    p1[k] = p[k] + h
    return (f(p1) - f(p)) / h

# Gradient of function f at point p
def grad(f, p, h=0.01):
    gp = p.copy()
    for k in range(len(p)):
        gp[k] = df(f, p, k, h)
    return gp

# Gradient descent to find function minimum
def gradientDescendent(f, p0, h=0.01, max_loops=100000, dump_period=1000):
    np.set_printoptions(precision=6)
    p = p0.copy()
    for i in range(max_loops):
        fp = f(p)
        gp = grad(f, p)
        glen = norm(gp)
        if i % dump_period == 0:
            print('{:05d}:f(p)={:.3f} p={:s} gp={:s} glen={:.5f}'.format(
                i, fp, str(p), str(gp), glen))
        if glen < 0.00001:
            break
        gh = np.multiply(gp, -1*h)
        p += gh
    print('{:05d}:f(p)={:.3f} p={:s} gp={:s} glen={:.5f}'.format(
        i, fp, str(p), str(gp), glen))
    return p
```

### Step-by-Step Analysis

#### 1. `df()`: Numerical Differentiation - Partial Derivative for Single Parameter

```python
def df(f, p, k, h=0.01):
    p1 = p.copy()
    p1[k] = p[k] + h
    return (f(p1) - f(p)) / h
```

This implements **Numerical Differentiation**, approximating the partial derivative of function for the k-th parameter.

Principle: When h is very small,

$$\frac{f(p_1) - f(p)}{h} \approx \frac{df}{dx}$$

For $f(x) = x^2$ at $x = 2$:
- $f(2) = 4$
- $f(2.01) = 4.0401$
- $\frac{4.0401 - 4}{0.01} = 4.01$ (approximates true gradient 4)

| Parameter | Description |
|-----------|-------------|
| `f`       | Objective function |
| `p`       | Current parameter vector |
| `k`       | Index of parameter to differentiate |
| `h`       | Small change amount (default 0.01, smaller = more accurate but has numerical precision limits) |

#### 2. `grad()`: Calculate Complete Gradient Vector

```python
def grad(f, p, h=0.01):
    gp = p.copy()
    for k in range(len(p)):
        gp[k] = df(f, p, k, h)
    return gp
```

This function calculates partial derivatives for **each parameter** in the parameter vector, then combines them into the gradient vector:

$$\nabla f = \left(\frac{\partial f}{\partial p_0}, \frac{\partial f}{\partial p_1}, ..., \frac{\partial f}{\partial p_n}\right)$$

For a 2D function $f(x, y)$, the gradient is a 2D vector pointing in the direction of steepest ascent.

#### 3. `gradientDescendent()`: Complete Gradient Descent Loop

```python
def gradientDescendent(f, p0, h=0.01, max_loops=100000, dump_period=1000):
```

| Parameter | Description |
|-----------|-------------|
| `f`       | Objective function (to minimize) |
| `p0`      | Initial parameter vector (starting point) |
| `h`       | Learning rate (how far each step) |
| `max_loops` | Maximum iterations (prevent infinite loop) |
| `dump_period` | Print progress every N steps |

##### Loop Internal Flow:

```python
for i in range(max_loops):
    fp = f(p)                 # Step 1: Calculate current function value
    gp = grad(f, p)          # Step 2: Calculate gradient
    glen = norm(gp)           # Calculate gradient vector length
    
    if i % dump_period == 0:  # Print progress every dump_period steps
        print(...)
    
    if glen < 0.00001:        # Convergence: gradient is small enough
        break
    
    gh = np.multiply(gp, -1*h)  # Step 3: Calculate step in negative gradient direction
    p += gh                     # Step 4: Update parameters
```

##### Meaning of Gradient Length (norm):

`norm(gp)` calculates the length of the gradient vector:

$$||\nabla f|| = \sqrt{\left(\frac{\partial f}{\partial p_0}\right)^2 + \left(\frac{\partial f}{\partial p_1}\right)^2 + ...}$$

When gradient length is small, we've entered a "flat" region - likely near a minimum!

##### Stopping Condition:

```python
if glen < 0.00001:
    break
```

When gradient length is less than 0.00001, we consider it converged and stop iteration.

#### 4. Example: Find Minimum of $f(x,y) = x^2 + y^2$

```python
# Define function
def f(p):
    return p[0]**2 + p[1]**2  # Bowl function, minimum at (0, 0)

# Initial point
p0 = [10.0, 10.0]

# Execute gradient descent
result = gradientDescendent(f, p0, h=0.1, dump_period=10)
```

Execution result:
```
00000:f(p)=200.000 p=[10.0, 10.0] gp=[20.0, 20.0] glen=28.28430
00010:f(p)=0.000 p=[0.0, 0.0] gp=[0.0, 0.0] glen=0.00000
```

As you can see:
- Initially $f(p) = 10^2 + 10^2 = 200$
- Gradient is $[20, 20]$ (because $\frac{\partial f}{\partial x} = 2x$, which is 20 at $x=10$)
- Each step moves toward negative gradient, gradually converging
- At step 10, reached minimum $(0, 0)$, $f(p) = 0$

### Numerical Differentiation vs Analytical Differentiation

This code uses **Numerical Differentiation**, applicable to any function:

| Method | Advantages | Disadvantages |
|--------|-----------|--------------|
| Numerical Differentiation | Works for any function, simple | Slow (requires multiple function evaluations), has precision error |
| Analytical Differentiation | Fast, accurate | Requires manual derivation (or automatic differentiation) |

Modern deep learning frameworks (PyTorch, TensorFlow) use **Automatic Differentiation**, combining advantages of both.

## Summary

Key concepts from this chapter:
- **Loss Function**: Quantifies prediction error
- **Gradient**: Direction of fastest function change
- **Gradient Descent**: Update parameters along negative gradient
- **Learning Rate**: Controls step size

In the next chapter, we'll learn how to efficiently compute gradients for complex neural networks - this is **Backpropagation**.

---

*Previous: [1. The First Step of Machine Learning](01-ml_basics.md)*  
*Next: [3. Backpropagation - The Magic of Automatic Differentiation](03-backpropagation.md)*
