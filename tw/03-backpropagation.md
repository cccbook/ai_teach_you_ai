# 3. 反向傳播——自動微分的魔法

## 為什麼需要反向傳播？

上一章我們學了梯度下降：沿著負梯度方向更新參數，讓損失越來越小。

問題來了：**如何計算複雜神經網路的梯度？**

一個現代神經網路可能有：
- 上億個參數
- 數百層結構
- 複雜的非線性運算

不可能手動計算每個參數的梯度。這時，反向傳播（Backpropagation）登場了。

## 核心思想：鏈法則

反向傳播的數學基礎是**鏈法則**（Chain Rule）。

### 什麼是鏈法則？

如果 y 是 u 的函數，u 是 x 的函數：

$$y = f(u) \quad u = g(x)$$

那麼 y 對 x 的導數是：

$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$

簡單例子：

$$y = \sin(x^2)$$

令 $u = x^2$，則 $y = \sin(u)$

$$\frac{dy}{dx} = \cos(u) \cdot 2x = 2x \cos(x^2)$$

### 為什麼鏈法則這麼重要？

因為神經網路就是一個長長的函數鏈：

$$loss = f_4(f_3(f_2(f_1(x))))$$

應用鏈法則：

$$\frac{\partial loss}{\partial x} = \frac{\partial f_4}{\partial f_3} \cdot \frac{\partial f_3}{\partial f_2} \cdot \frac{\partial f_2}{\partial f_1} \cdot \frac{\partial f_1}{\partial x}$$

從輸出往輸入方向，一層層「傳回去」——這就是「反向」的由來。

## 前向傳遞 vs 反向傳遞

```
前向傳遞（Forward Pass）：
輸入 → 層1 → 層2 → 層3 → 輸出 → 計算損失

反向傳遞（Backward Pass）：
損失 ← 層3梯度 ← 層2梯度 ← 層1梯度 ← 輸入梯度
```

### 前向傳遞

資料從輸入流到輸出，每層做運算產生中間結果。

```python
# 簡化的神經網路
h1 = x * w1 + b1      # 第一層
h2 = relu(h1)         # 啟動函數
out = h2 * w2 + b2     # 輸出層
loss = (out - y)**2   # 計算損失
```

### 反向傳遞

從損失函數開始，用鏈法則一層層往回算梯度。

```python
# 反向傳遞
d_out = 2 * (out - y)           # ∂loss/∂out
d_w2 = d_out * h2                # ∂loss/∂w2
d_b2 = d_out                     # ∂loss/∂b2
d_h2 = d_out * w2                # ∂loss/∂h2
d_relu = d_h2 * (h1 > 0)         # ∂relu/∂h1
d_w1 = d_relu * x                # ∂loss/∂w1
d_b1 = d_relu                    # ∂loss/∂b1
```

## 計算圖：視覺化理解

用計算圖來看更清楚：

```
       x ──┬── w1 ──┬── relu ──┬── w2 ──┬── out ──┬── loss
            │        │          │        │         │
           2         3          4        5         6
```

### 前向傳遞（紅色箭頭）：
```
x=1 → 乘以 w1=2 → 得 2 → relu(2)=2 → 乘以 w2=3 → 得 6 → 計算 loss
```

### 反向傳遞（藍色箭頭）：
```
∂loss/∂out=1 → 乘以 w2=3 → 得 ∂loss/∂relu=3 → relu導數=1 → 得 ∂loss/∂(wx+b)=3 → 乘以 x=1 → 得 ∂loss/∂w1=3
```

## 自動微分：讓電腦自動算梯度

手動算梯度太痛苦了，而且容易出錯。

**自動微分**（Automatic Differentiation）讓電腦自動做這件事。

原理：
1. 把運算拆成基本操作（加、減、乘、除、relu、log、exp...）
2. 每個基本操作定義好自己的梯度
3. 前向傳遞時記錄計算圖
4. 反向傳遞時自動套用鏈法則

這就是 `micrograd` 和 PyTorch 等框架的核心原理。

## 反向傳播的關鍵運算

### 矩陣乘法

```python
# 前向：y = W @ x
y = W @ x

# 反向：根據上游梯度和 W^T 計算
d_x = W.T @ d_y
d_W = d_y @ x.T
```

### ReLU 啟動函數

```python
def relu(x):
    return max(0, x)

# 前向：y = relu(x)
y = max(0, x)

# 反向：
# 如果 x > 0：dy/dx = 1，所以上游梯度直接傳下去
# 如果 x <= 0：dy/dx = 0，所以梯度是 0
d_x = d_y * (x > 0)
```

### Softmax 函數

```python
# 前向
exp_x = exp(x - max(x))  # 數值穩定化
y = exp_x / sum(exp_x)

# 反向（複雜一些）
# 這裡先跳過詳細推導，第六章 microgpt 會再提到
```

## 為什麼 GPU 擅長這個？

神經網路訓練需要大量矩陣運算——正是 GPU 的強項：

| 特性 | CPU | GPU |
|-----|-----|-----|
| 核心數 | 幾個到幾十個 | 幾千個 |
| 設計目標 | 複雜邏輯 | 大量並行運算 |
| 矩陣乘法速度 | 慢 | 快 100-1000 倍 |

神經網路的梯度計算高度並行，正好發揮 GPU 的優勢。

## 總結

這一章的核心概念：
- **鏈法則**：計算複合函數導數的核心數學工具
- **反向傳播**：沿著計算圖從後往前計算梯度
- **前向傳遞**：資料從輸入流到輸出
- **自動微分**：電腦自動計算複雜函數的梯度

下一章，我們會用 micrograd 這個微型框架實際動手做，徹底理解自動微分！

---

*上一步：[2. 梯度與梯度下降](02-gradient_descent.md)*  
*下一步：[4. 用 micrograd 從零理解自動微分](04-micrograd.md)*
