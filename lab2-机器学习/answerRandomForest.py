from numpy.random import rand
import mnist
from answerTree import *
import numpy as np

# 超参数
# TODO: You can change the hyperparameters here
num_tree = 100     # 树的数量
ratio_data = 0.6   # 采样的数据比例
ratio_feat = 0.3 # 采样的特征比例
hyperparams = {
    "depth":15, 
    "purity_bound":0.2,
    "gainfunc": gain
    } # 每颗树的超参数



def buildtrees(X, Y):
    """
    构建随机森林
    @param X: n*d, 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @param Y: n, 样本的label
    @return: List of DecisionTrees, 随机森林
    """
    # TODO: YOUR CODE HERE
    # 提示：整体流程包括样本扰动、属性扰动和预测输出

    n,d=X.shape
    k=int(n*ratio_data)
    l=int(d*ratio_feat)

    Trees=[]

    for i in range(num_tree):
        row_filter = np.array([True] * k + [False] * (n - k))
        np.random.shuffle(row_filter)
        col_filter = np.array([True] * l + [False] * (d - l))
        np.random.shuffle(col_filter)
        childX=X[row_filter]
        childY=Y[row_filter]

        child_feat=[]
        for j in range(d):
            if(col_filter[j]==True):
                child_feat.append(j)

        Trees.append( buildTree(childX,childY,child_feat,hyperparams["depth"],hyperparams["purity_bound"],hyperparams["gainfunc"],prefixstr="",) )

    return Trees
    raise NotImplementedError    

def infertrees(trees, X):
    """
    随机森林预测
    @param trees: 随机森林
    @param X: n*d, 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @return: n, 预测的label
    """
    pred = [inferTree(tree, X)  for tree in trees]
    pred = list(filter(lambda x: not np.isnan(x), pred))
    upred, ucnt = np.unique(pred, return_counts=True)
    return upred[np.argmax(ucnt)]
