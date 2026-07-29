# 北京大学《人工智能引论》实验记录（2026 春）

这里整理了我在《人工智能引论》课上完成的四次实验。代码基本保留提交时的样子，README 主要用来说明每次实验做了什么，以及我在实现时遇到的问题。

| 实验 | 内容 |
|---|---|
| [Lab 1](./Lab1-Search-and-Game-Playing/) | DFS、BFS、UCS、A*、Minimax、Alpha-Beta、MCTS |
| [Lab 2](./Lab2-Machine-Learning-from-Scratch/) | 逻辑回归、决策树、随机森林、自动微分、MLP、CNN |
| [Lab 3](./Lab3-NLP-and-LLM-Agent/) | 朴素贝叶斯、Attention、TF-IDF 问答、LLM Agent |
| [Lab 4](./Lab4-Robotics-and-Simulation/) | 粒子滤波、PD 控制、RRT 路径规划 |

## 各次实验

### Lab 1：搜索与博弈

第一部分是在 Pacman 里实现几种经典搜索算法。我分别用栈、队列和优先队列完成 DFS、BFS、UCS 和 A*，还为“走遍四个角”设计了启发函数。第二部分是多智能体搜索，包括 Minimax、Alpha-Beta 剪枝和 MCTS。

这次实验让我比较清楚地理解了：搜索算法的区别不只是换一个数据结构，还包括节点什么时候判重、路径代价怎么维护，以及启发函数怎样在不影响正确性的情况下减少搜索量。

### Lab 2：从零实现机器学习

这次实验的内容最多。前面是逻辑回归、决策树和随机森林，后面需要自己实现计算图和反向传播，再用这些模块训练 MLP 和 CNN。

我补全了 Linear、激活函数、BatchNorm、Dropout、Softmax、卷积和最大池化等节点。最后的 CNN 还加入了平移、旋转和缩放的数据增强。写完以后，我对反向传播、张量维度和训练/推理模式的区别熟悉了很多。

### Lab 3：NLP 与 LLM

这一部分先做了朴素贝叶斯情感分类和 TF-IDF 检索问答，然后使用词向量与 Attention 做文本分类，最后让大语言模型根据迷宫状态选择动作。

我觉得比较有意思的是，传统方法和 LLM 并不是完全割裂的。TF-IDF 可以看成最基础的检索模块，而 LLM Agent 除了调用模型，还要把状态描述清楚、限制输出格式，并检查动作是否合法。

### Lab 4：定位、控制与规划

这次实验把粒子滤波、PD 控制和 RRT 放在同一个物理仿真环境中。粒子滤波估计位置，RRT 负责找路，PD 控制器负责让 Pacman 沿路径移动。

我的 RRT 使用双向扩展，并加入了目标偏置、路径平滑、卡住检测和重新规划。实际调试时，最麻烦的地方不是“能不能找到路径”，而是路径能不能被控制器稳定跟踪，尤其是靠近墙角时很容易卡住。

## 仓库结构

```text
.
├── Lab1-Search-and-Game-Playing/
├── Lab2-Machine-Learning-from-Scratch/
├── Lab3-NLP-and-LLM-Agent/
└── Lab4-Robotics-and-Simulation/
```

每个目录里都有对应的原始作答源码和更详细的说明。

## 说明

- 没有上传课程数据集、评测程序、模型权重或 API Key。
- 部分代码依赖课程框架，不能直接单独运行。
- 代码仅用于学习记录，请不要直接作为课程作业提交。
