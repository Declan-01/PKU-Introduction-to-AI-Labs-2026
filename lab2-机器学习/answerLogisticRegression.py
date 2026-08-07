import numpy as np

# 超参数
# TODO: You can change the hyperparameters here
lr = 1e-1+2*1e-2  # 学习率
wd = 1e-2  # l2正则化项系数

EPS=1e-8

def predict(X, weight, bias):
    """
    使用输入的weight和bias，预测样本X是否为数字0。
    @param X: (n, d) 每行是一个输入样本。n: 样本数量, d: 样本的维度
    @param weight: (d,)
    @param bias: (1,)
    @return: (n,) 线性模型的输出，即wx+b
    """
    # TODO: YOUR CODE HERE

    return X@weight+bias


    raise NotImplementedError

def sigmoid(x):
    return 1 / (np.exp(-x) + 1)


def step(X, weight, bias, Y):
    """
    单步训练, 进行一次forward、backward和参数更新
    @param X: (n, d) 每行是一个训练样本。 n: 样本数量， d: 样本的维度
    @param weight: (d,)
    @param bias: (1,)
    @param Y: (n,) 样本的label, 1表示为数字0, -1表示不为数字0
    @return:
        haty: (n,) 模型的输出, 为正表示数字为0, 为负表示数字不为0
        loss: (1,) 由交叉熵损失函数计算得到
        weight: (d,) 更新后的weight参数
        bias: (1,) 更新后的bias参数
    """
    # 用法参考：haty, loss, weight, bias = step(X, weight, bias, Y)
    # 结论： p(y|x;w;b)=sigmoid(y*f(x))

    n,d=X.shape

    haty=X@weight+bias

    p=sigmoid(haty*Y)

    loss_vector=-np.log(p+EPS)

    loss=loss_vector.mean()
    
    gradient_haty = -(1 - p) * Y
    
    gradient_w = ((X.T @ gradient_haty))/n+2*wd*weight

    gradient_bias=gradient_haty.mean()

    weight-=lr*gradient_w

    bias-=lr*gradient_bias

    return [haty,loss,weight,bias]

    # TODO: YOUR CODE HERE
    raise NotImplementedError
