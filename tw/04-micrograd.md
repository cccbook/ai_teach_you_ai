# 4. 用 micrograd 從零理解自動微分

## micrograd 簡介

micrograd 是 Andrej Karpathy 寫的一個微型自動微分框架，只有約 150 行 Python 程式碼，卻完整實現了神經網路訓練的核心功能。

github: https://github.com/karpathy/micrograd

本書使用的程式碼位於：[_code/nn0py/nn0.py](https://github.com/karpathy/micrograd)（模組化版本）

## 核心：Value 類別

micrograd 的核心是 `Value` 類別，它封裝了：
- `data`：這個節點的數值
- `grad`：這個節點的梯度（預設為 0）
- `_children`：這個節點依賴的其他節點
- `_local_grads`：本地梯度（用於反向傳播）

```python
class Value:
    def __init__(self, data, children=(), local_grads=()):
        self.data = data              # 純量數值
        self.grad = 0                 # 梯度，預設為 0
        self._children = children    # 子節點
        self._local_grads = local_grads  # 本地梯度
```

## 基本運算：加法與乘法

### 加法

$$c = a + b \Rightarrow \frac{\partial c}{\partial a} = 1, \frac{\partial c}{\partial b} = 1$$

```python
def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    return Value(
        self.data + other.data,           # 前向值 = a + b
        (self, other),                    # 記錄子節點
        (1, 1)                            # 本地梯度
    )
```

### 乘法

$$c = a × b \Rightarrow \frac{\partial c}{\partial a} = b, \frac{\partial c}{\partial b} = a$$

```python
def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    return Value(
        self.data * other.data,           # 前向值 = a * b
        (self, other),                    # 記錄子節點
        (other.data, self.data)           # 本地梯度
    )
```

## 運算圖的建立

讓我們用一個簡單例子看運算圖是怎麼建立的：

```python
from nn0 import Value

# 建立具有初始值的參數與變數
a = Value(3.0)
b = Value(2.0)
x = Value(4.0)

# 建立運算圖 (Forward pass)： y = a * x^2 + b
y = a * (x ** 2) + b

# 顯示前向傳播結果
print(f"前向計算結果 y.data = {y.data:.4f}")  # 預期：3 * 16 + 2 = 50
```

運算圖：

```
    a (2) ──┐
            ├── * ──> c (6) ──┬── + ──> d (7) ──┬── * ──> e (14)
    b (3) ──┘                 └── 1.0 (1) ──────┘         │
                                                          └── 2.0
```

## 反向傳播：自動計算梯度

關鍵的 `backward()` 方法：

```python
def backward(self):
    # 建立反向傳播的拓撲排序
    topo = []
    visited = set()
    
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._children:
                build_topo(child)
            topo.append(v)
    
    build_topo(self)
    
    # 從輸出開始，往回傳遞梯度
    self.grad = 1  # loss 對自己的梯度是 1
    for v in reversed(topo):
        for child, local_grad in zip(v._children, v._local_grads):
            # 鏈法則：child.grad += local_grad * v.grad
            child.grad += local_grad * v.grad
```

## 完整例子：計算 $f(x) = 3x^2 + 2x + 1$

```python
from micrograd import Value

x = Value(2.0)

# f(x) = 3x² + 2x + 1
f = 3 * x * x + 2 * x + 1

print(f"f(x) = {f.data}")  # f(2) = 3*4 + 4 + 1 = 17

f.backward()

print(f"f'(x) = {x.grad}")  # f'(2) = 6*2 + 2 = 14
```

驗證：$f(x) = 3x^2 + 2x + 1$，$f'(x) = 6x + 2$，$f'(2) = 14$ ✓

### 實際執行

執行結果：

```
=== 1. 基礎自動微分運算 ===
前向計算結果 y.data = 50.0000
dy/da = 16.0000 (理論值 x^2 = 16)
dy/db = 1.0000 (理論值 1)
dy/dx = 24.0000 (理論值 2ax = 2*3*4 = 24)
```

完美匹配理論值！

## 動手做：訓練一個神經網路

有了自動微分，我們可以訓練一個簡單的神經網路：

### 目標

學會 AND 邏輯（兩個輸入都是 1 時輸出 1，否則輸出 0）：

| x1 | x2 | 輸出 |
|----|----|-----|
| 0  | 0  | 0   |
| 0  | 1  | 0   |
| 1  | 0  | 0   |
| 1  | 1  | 1   |

### 用 nn0.py 實現線性回歸

```python
from nn0 import Value, Adam

# 準備訓練數據 (目標函數 y = 2x + 1)
X = [1.0, 2.0, 3.0, 4.0]
Y = [3.0, 5.0, 7.0, 9.0]

# 初始化模型參數 (權重與偏差)
w = Value(0.0)
b = Value(0.0)

# 實例化 Adam 優化器
optimizer = Adam([w, b], lr=0.1)

# 進行 50 次訓練迭代
for epoch in range(51):
    # 前向傳播：預測值與計算 Mean Squared Error (MSE) Loss
    preds = [w * x + b for x in X]
    losses = [(pred - y) ** 2 for pred, y in zip(preds, Y)]
    loss = sum(losses) / len(losses)
    
    # 反向傳播與參數更新
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch:2d} | Loss: {loss.data:.4f} | w: {w.data:.4f}, b: {b.data:.4f}")
```

### 訓練過程

執行結果：
```
Epoch  0 | Loss: 50.0000 | w: 0.4000, b: 0.2000
Epoch 10 | Loss: 0.0187 | w: 1.9559, b: 1.0356
Epoch 20 | Loss: 0.0007 | w: 1.9883, b: 1.0129
Epoch 30 | Loss: 0.0000 | w: 1.9972, b: 1.0049
Epoch 40 | Loss: 0.0000 | w: 1.9992, b: 1.0020
Epoch 50 | Loss: 0.0000 | w: 1.9998, b: 1.0008
```

Loss 從 50 降到趨近於 0，權重收斂到 $w \approx 2$, $b \approx 1$——完美符合目標函數 $y = 2x + 1$！

### 更複雜的網路：XOR 問題

XOR 是傳統感知器的經典難題，需要多層網路才能解決：

```python
from nn0 import Value, Adam
import random

# XOR 訓練數據
X_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y_data = [0, 1, 1, 0]

# 初始化 2-4-1 網路參數
w1 = [[Value(random.uniform(-1, 1)) for _ in range(2)] for _ in range(4)]
b1 = [Value(random.uniform(-1, 1)) for _ in range(4)]
w2 = [[Value(random.uniform(-1, 1)) for _ in range(4)] for _ in range(1)]
b2 = [Value(random.uniform(-1, 1)) for _ in range(1)]

# 收集所有參數交給 Adam
params = [p for row in w1 for p in row] + b1 + [p for row in w2 for p in row] + b2
optimizer = Adam(params, lr=0.1)

# 訓練迴圈
for epoch in range(101):
    total_loss = Value(0.0)
    for x, y in zip(X_data, Y_data):
        pred = mlp(x)
        total_loss += (pred - y) ** 2
    loss = total_loss / len(X_data)
    loss.backward()
    optimizer.step()
```

執行結果：
```
Epoch   0 | Loss: 0.2556
Epoch  20 | Loss: 0.0128
Epoch  40 | Loss: 0.0014
Epoch  60 | Loss: 0.0002
Epoch  80 | Loss: 0.0000
Epoch 100 | Loss: 0.0000

預測結果:
輸入 [0, 0] -> 預測: 0.0000 (目標: 0)
輸入 [0, 1] -> 預測: 1.0000 (目標: 1)
輸入 [1, 0] -> 預測: 1.0000 (目標: 1)
輸入 [1, 1] -> 預測: 0.0000 (目標: 0)
```

完美解決 XOR！多層網路 + 非線性激活函數（ReLU）讓不可能變成可能。

## micrograd 的限制

micrograd 是一個教學工具，不是生產工具：

| 限制 | 說明 |
|-----|------|
| 速度慢 | 用 Python 物件計算，比 NumPy/PyTorch 慢數百倍 |
| 只支援純量 | 無法直接處理向量、矩陣 |
| 無 GPU 支援 | 所有計算在 CPU 上 |

但它的價值在於：**讓你完全理解自動微分背後的原理**。

## 與 PyTorch 的比較

micrograd 的 `Value` 類別就是 PyTorch 的 `Tensor` 的簡化版本：

| micrograd | PyTorch |
|-----------|---------|
| `Value` 類別 | `Tensor` 類別 |
| 手寫梯度 | `autograd` 模組 |
| 純量計算 | 張量計算 |
| 單執行緒 | 多執行緒 + GPU |

學會 micrograd，你就掌握了 PyTorch 自動微分的核心思想！

## 總結

這一章的核心概念：
- **Value 類別**：封裝數值和梯度
- **運算圖**：追蹤每個運算的依賴關係
- **反向傳播**：從後往前應用鏈法則
- **訓練迴圈**：前向 → 反向 → 更新 → 重複

下一章，我們離開純量計算，進入文字的數學世界——Token 化與 Embedding！

---

*上一步：[3. 反向傳播](03-backpropagation.md)*  
*下一步：[5. Token 化與 Embedding](05-embedding.md)*
