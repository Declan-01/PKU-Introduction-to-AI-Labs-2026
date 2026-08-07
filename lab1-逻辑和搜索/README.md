### part1:搜索
q1,q2,q3要求实现三种不同的搜索算法，q4则要求自己设计一个启发函数

q1/2/3按照PPT中的伪代码以及readme中的框架提示不难实现，主要关注**节点是否应该进入列表**（通过visited实现），**节点何时出列表**（通过选择数据类型实现），注意通过autograder的反馈调整顺序

如果对这三种搜索算法有不清楚之处，建议参考[cs188在search部分的讲义](https://inst.eecs.berkeley.edu/~cs188/textbook/search/uninformed.html)

q4则要求自己设计距离启发函数，这里建议自己想着玩，还是挺有意思的

---
实际得分：12/12
---
### part2：Agent
q1要求实现minimax,q2则要求进一步实现alpha-beta pruning，按照PPT中伪代码思路来写即可

q3则要求实现MCTS，主要思路是：

1. selection:选择要进行expand的节点，其**选择标准**是：如果仍有尚未探索的子节点，就是自己；如果所有子节点都探索过了，就根据 *UCT* 选择一个子节点
2. expansion:对未完全拓展的节点进行拓展,在这个lab中直接进行全子节点拓展
3. simulation:这里直接用*启发函数*即可
4. back-propagation:用模拟结果更新每个节点的属性


需要注意的是，节点的*denominator*需要初始化为1，否则在*expansion*环节子节点被创建后，直接进入*simulation*环节时会出现除以0的情况

---
实际得分：12/12
