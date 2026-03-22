# 1. The First Step of Machine Learning - From Prediction to Learning

## What is Machine Learning?

Machine learning is the process of teaching computers to learn patterns from data, then use those patterns to make predictions or decisions. This is different from traditional programming:

| Traditional Programming | Machine Learning |
|----------------------|-----------------|
| Rules → Input → Output | Input → Output (Data) → Rules |
| Humans write explicit rules | Machine automatically finds rules |

## A Simple Prediction Problem

Imagine you want to predict house prices. Given this data:

| Size (ping) | Price (10K) |
|-------------|------------|
| 20          | 400        |
| 35          | 700        |
| 50          | 1000       |
| 80          | 1600       |

Can you spot the pattern?

$$Price ≈ Size × 20$$

This is exactly what machine learning does: find patterns from data.

## Basic Workflow of Supervised Learning

```
Input Data → Feature Extraction → Model Prediction → Compare with Answer → Adjust Model → Predict
```

### 1. Input and Output

- **Input (X)**: Information used for prediction (size, age, images...)
- **Output (y)**: Target to predict (price, category, next word...)

### 2. Model

A model is a mathematical function that transforms input to output:

```
y = f(x)
```

A simple linear model looks like this:

$$y = w × x + b$$

- `w`: Weight, determines the importance of input
- `b`: Bias, adjusts the baseline of output

### 3. Loss Function

After prediction, we need to know "how wrong we were":

$$L = (predicted - actual)^2$$

This is called Mean Squared Error (MSE). Lower loss means more accurate prediction.

## From "Can't" to "Can"

Initially, the model's predictions are bad:

```
Actual: 400   Predicted: 50    Loss: 122500
Actual: 700   Predicted: 120   Loss: 336400
Actual: 1000  Predicted: 200   Loss: 640000
```

After continuous adjustments (called "training"), predictions become more accurate:

```
Actual: 400   Predicted: 398   Loss: 4
Actual: 700   Predicted: 702   Loss: 4
Actual: 1000  Predicted: 998   Loss: 4
```

## AI Models Are More Complex Functions

Linear models are too simple for complex problems.

Modern AI uses **Neural Networks**, which are essentially complex functions:

$$y = f_{neural\ network}(x)$$

This function has:
- Extremely many parameters (millions to hundreds of billions)
- Multi-layer structure (origin of "deep" learning)
- Non-linear operations (able to learn complex patterns)

## Summary

Key concepts from this chapter:
- **Machine Learning**: Learning patterns from data
- **Supervised Learning**: Input → Model → Output, compare with correct answer
- **Model**: A complex mathematical function
- **Loss Function**: Quantifies prediction error

In the next chapter, we'll dive deep into how machines "learn" - how to adjust parameters to lower the loss.

---

*Next: [2. Learning from Mistakes - Gradient and Gradient Descent](02-gradient_descent.md)*
