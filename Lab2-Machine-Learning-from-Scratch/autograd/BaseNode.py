from typing import List
import math
import numpy as np
import numpy as np
from .Init import * 

EPS=1e-8

def shape(X):
    if isinstance(X, np.ndarray):
        ret = "ndarray"
        if np.any(np.isposinf(X)):
            ret += "_posinf"
        if np.any(np.isneginf(X)):
            ret += "_neginf"
        if np.any(np.isnan(X)):
            ret += "_nan"
        return f" {X.shape} "
    if isinstance(X, int):
        return "int"
    if isinstance(X, float):
        ret = "float"
        if np.any(np.isposinf(X)):
            ret += "_posinf"
        if np.any(np.isneginf(X)):
            ret += "_neginf"
        if np.any(np.isnan(X)):
            ret += "_nan"
        return ret
    else:
        raise NotImplementedError(f"unsupported type {type(X)}")

class Node(object):
    def __init__(self, name, *params):
        self.grad = [] # 节点的梯度，self.grad[i]对应self.params[i]在反向传播时的梯度
        self.cache = [] # 节点保存的临时数据
        self.name = name # 节点的名字
        self.params = list(params) # 用于Linear节点中存储weight和bias参数使用

    def num_params(self):
        return len(self.params)

    def cal(self, X):
        '''
        计算函数值。请在其子类中完成具体实现。
        '''
        pass

    def backcal(self, grad):
        '''
        计算梯度。请在其子类中完成具体实现。
        '''
        pass

    def flush(self):
        '''
        初始化或刷新节点内部数据，包括梯度和缓存
        '''
        self.grad = []
        self.cache = []

    def forward(self, X, debug=False):
        '''
        正向传播。输入X，输出正向传播的计算结果。
        '''
        if debug:
            print(self.name, shape(X))
        ret = self.cal(X)
        if debug:
            print(shape(ret))
        return ret

    def backward(self, grad, debug=False):
        '''
        反向传播。输入grad（该grad为反向传播到该节点的梯度），输出反向传播到下一层的梯度。
        '''
        if debug:
            print(self.name, shape(grad))
        ret = self.backcal(grad)
        if debug:
            print(shape(ret))
        return ret
    
    def eval(self):
        pass

    def train(self):
        pass


class relu(Node):
    # input X: (*)，即可能是任意维度
    # output relu(X): (*)
    def __init__(self):
        super().__init__("relu")

    def cal(self, X):
        self.cache.append(X)
        return np.clip(X, 0, None)

    def backcal(self, grad):
        return np.multiply(grad, self.cache[-1] > 0) 

class sigmoid(Node):
    # input X: (*)，即可能是任意维度
    # output sigmoid(X): (*)
    def __init__(self):
        super().__init__("sigmoid")

    def cal(self, X):
        # TODO: YOUR CODE HERE
        self.cache.append(X)
        return 1 / (np.exp(-X) + 1)
        raise NotImplementedError        

    def backcal(self, grad):
        # TODO: YOUR CODE HERE
        X=self.cache[-1]
        local_grad=(np.exp(-X))/np.square(1+np.exp(-X))

        return local_grad*grad
        raise NotImplementedError        
    
class tanh(Node):
    # input X: (*)，即可能是任意维度
    # output tanh(X): (*)
    def __init__(self):
        super().__init__("tanh")

    def cal(self, X):
        ret = np.tanh(X)
        self.cache.append(ret)
        return ret


    def backcal(self, grad):
        return np.multiply(grad, np.multiply(1+self.cache[-1], 1-self.cache[-1]))
    

class Linear(Node):
    # input X: (*,d1)
    # param weight: (d1, d2)
    # param bias: (d2)
    # output Linear(X): (*, d2)
    def __init__(self, indim, outdim):
        """
        初始化
        @param indim: 输入维度
        @param outdim: 输出维度
        """
        weight = kaiming_uniform(indim, outdim)
        bias = zeros(outdim)
        super().__init__("linear", weight, bias)

    def cal(self, X):
        # TODO: YOUR CODE HERE
        weight=self.params[0]
        bias=self.params[1]
        self.cache.append(X)
        return X@weight+bias
        raise NotImplementedError

    def backcal(self, grad):
        '''
        需要保存weight和bias的梯度，可以参考Node类和BatchNorm类
        '''
        # TODO: YOUR CODE HERE
        W=self.params[0]
        X=self.cache[-1]
        indim = W.shape[0]
        outdim = W.shape[1]

        X_2d = X.reshape(-1, indim)
        grad_2d = grad.reshape(-1, outdim)

        self.grad.append(X_2d.T @ grad_2d)
        self.grad.append(grad_2d.sum(axis=0))
        return grad @ W.T
        raise NotImplementedError


class StdScaler(Node):
    '''
    input shape (*)
    output (*)
    '''
    EPS = 1e-3
    def __init__(self, mean, std):
        super().__init__("StdScaler")
        self.mean = mean
        self.std = std

    def cal(self, X):
        X = X.copy()
        X -= self.mean
        X /= (self.std + self.EPS)
        return X

    def backcal(self, grad):
        return grad/ (self.std + self.EPS)
    


class BatchNorm(Node):
    '''
    input shape (*)
    output (*)
    '''
    EPS = 1e-8
    def __init__(self, indim, momentum: float = 0.9):
        super().__init__("batchnorm", ones((indim)), zeros(indim))
        self.momentum = momentum
        self.mean = None
        self.std = None
        self.updatemean = True
        self.indim = indim

    def cal(self, X):
        if self.updatemean:
            tmean, tstd = np.mean(X, axis=0, keepdims=True), np.std(X, axis=0, keepdims=True)
            if self.mean is None or self.std is None:
                self.mean = tmean
                self.std = tstd
            else:
                self.mean *= self.momentum
                self.mean += (1-self.momentum) * tmean
                self.std *= self.momentum
                self.std += (1-self.momentum) * tstd
        X = X.copy()
        X -= self.mean
        X /= (self.std + self.EPS)
        self.cache.append(X.copy())
        X *= self.params[0]
        X += self.params[1]
        return X

    def backcal(self, grad):
        X = self.cache[-1]
        self.grad.append(np.multiply(X, grad).reshape(-1, self.indim).sum(axis=0))
        self.grad.append(grad.reshape(-1, self.indim).sum(axis=0))
        return (grad*self.params[0])/ (self.std + self.EPS)
    
    def eval(self):
        self.updatemean = False

    def train(self):
        self.updatemean = True


class Dropout(Node):
    '''
    input shape (*)
    output (*)
    '''
    def __init__(self, p: float = 0.1):
        super().__init__("dropout")
        assert 0<=p<=1, "p 是dropout 概率，必须在[0, 1]中"
        self.p = p
        self.dropout = True

    def cal(self, X):
        if self.dropout:
            X = X.copy()
            mask = np.random.rand(*X.shape) < self.p
            np.putmask(X, mask, 0)
            X = X * (1/(1-self.p))
            self.cache.append(mask)
        return X
    
    def backcal(self, grad):
        if self.dropout:
            grad = grad.copy()
            np.putmask(grad, self.cache[-1], 0)
            grad = grad * (1/(1-self.p))
        return grad
    
    def eval(self):
        self.dropout=False

    def train(self):
        self.dropout=True


class Softmax(Node):
    # input X: (*)
    # output softmax(X): (*), softmax at 'dim'
    def __init__(self, dim=-1):
        super().__init__("softmax")
        self.dim = dim

    def cal(self, X):
        X = X - np.max(X, axis=self.dim, keepdims=True)
        expX = np.exp(X)
        ret = expX / expX.sum(axis=self.dim, keepdims=True)
        self.cache.append(ret)
        return ret

    def backcal(self, grad):
        softmaxX = self.cache[-1]
        grad_p = np.multiply(grad, softmaxX)
        return grad_p - np.multiply(grad_p.sum(axis=self.dim, keepdims=True), softmaxX)


class LogSoftmax(Node):
    # input X: (*)
    # output logsoftmax(X): (*), logsoftmax at 'dim'
    def __init__(self, dim=-1):
        super().__init__("logsoftmax")
        self.dim = dim

    def cal(self, X):
        # TODO: YOUR CODE HERE
        self.cache.append(X)
        X_shift = X - np.max(X, axis=self.dim, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(X_shift), axis=self.dim, keepdims=True))
        ret = X_shift - log_sum_exp
        self.cache.append(ret)
        return ret
        raise NotImplementedError

    def backcal(self, grad):
        # TODO: YOUR CODE HERE
        softmaxX = np.exp(self.cache[-1])
        return  grad - softmaxX * np.sum(
            grad, axis=self.dim, keepdims=True
        )
        raise NotImplementedError




class NLLLoss(Node):
    '''
    negative log-likelihood 损失函数
    '''
    # shape X: (*, d), y: (*)
    # shape value: number 
    # 输入：X: (*) 个预测，每个预测是个d维向量，代表d个类别上分别的log概率。  y：(*) 个整数类别标签
    # 输出：NLL损失
    def __init__(self, y):
        """
        初始化
        @param y: n 样本的label
        """
        super().__init__("NLLLoss")
        self.y = y

    def cal(self, X):
        y = self.y
        self.cache.append(X)
        return - np.sum(
            np.take_along_axis(X, np.expand_dims(y, axis=-1), axis=-1))

    def backcal(self, grad):
        X, y = self.cache[-1], self.y
        ret = np.zeros_like(X)
        np.put_along_axis(ret, np.expand_dims(y, axis=-1), -1, axis=-1)
        return grad * ret



class CrossEntropyLoss(Node):
    '''
    多分类交叉熵损失函数，不同于课上讲的二分类。它与NLLLoss的区别仅在于后者输入log概率，前者输入概率。
    '''
    # shape X: (*, d), y: (*)
    # shape value: number 
    # 输入：X: (*) 个预测，每个预测是个d维向量，代表d个类别上分别的概率。  y：(*) 个整数类别标签
    # 输出：交叉熵损失
    def __init__(self, y):
        """
        初始化
        @param y: n 样本的label
        """
        super().__init__("CELoss")
        self.y = y

    def cal(self, X):
        # TODO: YOUR CODE HERE
        # 提示，可以对照NLLLoss的cal
        #尝试：不加EPS是否可行
        y = self.y
        log_X=np.log(X+EPS)
        self.cache.append(log_X)
        return - np.sum(
            np.take_along_axis(log_X, np.expand_dims(y, axis=-1), axis=-1))


        raise NotImplementedError

    def backcal(self, grad):
        # TODO: YOUR CODE HERE
        # 提示，可以对照NLLLoss的backcal
        X, y = self.cache[-1], self.y
        ret = np.zeros_like(X)
        np.put_along_axis(ret, np.expand_dims(y, axis=-1), -1, axis=-1)
        return grad * ret/np.exp(X)
        raise NotImplementedError
    
class InputNorm(Node):
    """X / 255.0 - 0.5 for 4D tensors (N,C,H,W)."""
    def __init__(self):
        super().__init__("inputnorm")
    def cal(self, X):
        return X / 255.0 - 0.5
    def backcal(self, grad):
        return grad / 255.0


class Conv2d(Node):
    """2D convolution with im2col forward and col2im backward."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        fan_in = in_channels * kernel_size * kernel_size
        weight = (3.0 / fan_in) ** 0.5 * (2 * np.random.rand(fan_in, out_channels) - 1)
        weight = weight.T.reshape(out_channels, in_channels, kernel_size, kernel_size)
        bias = np.zeros(out_channels)
        super().__init__("conv2d", weight, bias)

    def cal(self, X):
        weight = self.params[0]; bias = self.params[1]
        stride = self.stride; pad = self.padding; K = self.kernel_size
        N, C_in, H, W = X.shape; C_out = self.out_channels

        if pad > 0:
            X_pad = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
        else:
            X_pad = X

        H_out = (H + 2 * pad - K) // stride + 1
        W_out = (W + 2 * pad - K) // stride + 1

        shape = (N, C_in, H_out, W_out, K, K)
        strides = (X_pad.strides[0], X_pad.strides[1],
                   X_pad.strides[2] * stride, X_pad.strides[3] * stride,
                   X_pad.strides[2], X_pad.strides[3])
        patches = np.lib.stride_tricks.as_strided(X_pad, shape=shape, strides=strides)

        self.cache.append(X); self.cache.append(X_pad)
        cols = patches.copy().reshape(N * H_out * W_out, C_in * K * K)
        self.cache.append(cols)

        w_2d = weight.reshape(C_out, C_in * K * K).T
        out = cols @ w_2d
        out = out.reshape(N, H_out, W_out, C_out).transpose(0, 3, 1, 2)
        out = out + bias.reshape(1, C_out, 1, 1)
        return out

    def backcal(self, grad):
        weight = self.params[0]
        X_pad = self.cache[-2]; cols = self.cache[-1]
        N, C_out, H_out, W_out = grad.shape
        C_in = self.in_channels; K = self.kernel_size
        stride = self.stride; pad = self.padding

        grad_2d = grad.transpose(0, 2, 3, 1).reshape(-1, C_out)
        grad_w = cols.T @ grad_2d
        grad_w = grad_w.T.reshape(C_out, C_in, K, K)
        self.grad.append(grad_w)
        grad_b = grad.sum(axis=(0, 2, 3))
        self.grad.append(grad_b)

        w_2d = weight.reshape(C_out, C_in * K * K)
        grad_cols = grad_2d @ w_2d
        grad_cols_6d = grad_cols.reshape(N, H_out, W_out, C_in, K, K).transpose(0, 3, 1, 2, 4, 5)

        H_pad, W_pad = X_pad.shape[2], X_pad.shape[3]
        grad_X_pad = np.zeros((N, C_in, H_pad, W_pad))
        for ki in range(K):
            for kj in range(K):
                grad_X_pad[:, :, ki:ki + H_out * stride:stride, kj:kj + W_out * stride:stride] += \
                    grad_cols_6d[:, :, :, :, ki, kj]

        if pad > 0:
            return grad_X_pad[:, :, pad:-pad, pad:-pad]
        return grad_X_pad


class MaxPool2d(Node):
    """2D max pooling."""
    def __init__(self, kernel_size, stride=None):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        super().__init__("maxpool2d")

    def cal(self, X):
        K = self.kernel_size; stride = self.stride
        N, C, H, W = X.shape
        H_out = (H - K) // stride + 1
        W_out = (W - K) // stride + 1

        X_r = X.reshape(N, C, H_out, K, W_out, K)
        X_t = X_r.transpose(0, 1, 2, 4, 3, 5)
        patches = X_t.reshape(N, C, H_out, W_out, K * K)

        out = patches.max(axis=4)
        max_idx = patches.argmax(axis=4)
        self.cache.append(max_idx)
        self.cache.append((N, C, H_out, W_out, H, W))
        return out

    def backcal(self, grad):
        max_idx = self.cache[-2]
        N, C, H_out, W_out, H_in, W_in = self.cache[-1]
        K = self.kernel_size; stride = self.stride

        grad_in = np.zeros((N, C, H_in, W_in))
        ki = max_idx // K; kj = max_idx % K

        n_idx = np.arange(N).reshape(N, 1, 1, 1).repeat(C, 1).repeat(H_out, 2).repeat(W_out, 3).ravel()
        c_idx = np.arange(C).reshape(1, C, 1, 1).repeat(N, 0).repeat(H_out, 2).repeat(W_out, 3).ravel()
        h_idx = np.arange(H_out).reshape(1, 1, H_out, 1).repeat(N, 0).repeat(C, 1).repeat(W_out, 3).ravel()
        w_idx = np.arange(W_out).reshape(1, 1, 1, W_out).repeat(N, 0).repeat(C, 1).repeat(H_out, 2).ravel()

        in_h = (h_idx * stride + ki.ravel()).astype(int)
        in_w = (w_idx * stride + kj.ravel()).astype(int)

        np.add.at(grad_in, (n_idx, c_idx, in_h, in_w), grad.ravel())
        return grad_in


class Flatten(Node):
    """Flatten spatial dimensions: (N,C,H,W) -> (N,C*H*W)."""
    def __init__(self):
        super().__init__("flatten")

    def cal(self, X):
        self.cache.append(X.shape)
        return X.reshape(X.shape[0], -1)

    def backcal(self, grad):
        return grad.reshape(self.cache[-1])
