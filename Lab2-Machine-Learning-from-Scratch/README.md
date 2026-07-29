# Lab 2：从零实现机器学习

这次实验从逻辑回归和决策树开始，后半部分要自己写计算图、反向传播和神经网络。很多平时一行框架代码能完成的操作，这次都需要自己实现。

## Q1：逻辑回归

我用 NumPy 完成了二分类逻辑回归，包括前向计算、损失函数、梯度和 L2 正则化。

```text
z = Xw + b
p = sigmoid(y · z)
loss = -log(p)
```

代码使用矩阵运算一次处理一个 batch。计算对数时还要加一个很小的 `EPS`，不然概率接近 0 时会出现数值问题。

## Q2：决策树与随机森林

决策树里实现了信息熵、信息增益、信息增益率和 Gini 指数，并递归选择特征、划分数据。树停止生长后，叶节点用当前样本中的多数类别作为预测。

随机森林会为每棵树随机抽取一部分样本和特征，最后让多棵树投票。单棵树可能很容易受训练数据影响，而随机森林通过多棵不同的树让结果更稳定。

## Q3：计算图和 MLP

我补全了计算图中的前向传播、反向传播和参数更新，并实现了这些常用节点：

- Linear、ReLU、Sigmoid、Tanh；
- BatchNorm、LayerNorm、Dropout；
- Softmax、LogSoftmax、NLLLoss、CrossEntropy；
- L1 和 L2 正则化。

MLP 的结构比较简单：

```text
784 → Linear(256) → BatchNorm → ReLU
    → Linear(10) → LogSoftmax → NLLLoss
```

这一部分最容易出错的是梯度形状。一个节点的 forward 能跑通，并不代表 backward 就是对的，还要仔细处理 batch 维度、广播和保存的中间变量。

## Q4：CNN 与数据增强

在前面的计算图上，我继续实现了 Conv2d、MaxPool2d、Flatten 和 InputNorm。卷积用类似 `im2col` 的方法，把每个局部窗口展开后统一做矩阵乘法；最大池化在 forward 时记录最大值位置，backward 时再把梯度放回原来的位置。

最后使用的网络是：

```text
InputNorm
→ Conv2d(1, 8, 3, padding=1) → ReLU
→ MaxPool2d(2)
→ Flatten
→ Linear(8×14×14, 256) → ReLU → Dropout(0.2)
→ Linear(256, 128) → ReLU
→ Linear(128, 10) → LogSoftmax → NLLLoss
```

训练时加入了随机平移、旋转和缩放。MNIST 数字即使稍微移动或倾斜，类别也不会改变，所以这些增强可以让模型适应更多写法。

## 这次实验的收获

这次实验让我把课上学到的公式和实际代码对应了起来。尤其是手写反向传播后，我对 PyTorch 中计算图和 `backward()` 大致在做什么有了更具体的认识。调 CNN 时也发现，很多问题来自维度和训练细节，而不是模型结构本身。
