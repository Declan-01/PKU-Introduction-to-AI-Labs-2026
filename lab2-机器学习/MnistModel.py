import numpy as np
import modelLogisticRegression as LR
import modelTree as Tree
import modelRandomForest as Forest
import modelSoftmaxRegression as SR
import modelMultiLayerPerceptron as MLP
import pickle



class NullModel:

    def __init__(self):
        pass

    def __call__(self, figure):
        return 0


class LRModel:
    def __init__(self) -> None:
        with open(LR.save_path, "rb") as f:
            self.weight, self.bias = pickle.load(f)

    def __call__(self, figure):
        pred = figure @self.weight + self.bias
        return 0 if pred > 0 else 1

class TreeModel:
    def __init__(self) -> None:
        with open(Tree.save_path, "rb") as f:
            self.root = pickle.load(f)
    
    def __call__(self, figure):
        return Tree.inferTree(self.root, Tree.discretize(figure.flatten()))


class ForestModel:
    def __init__(self) -> None:
        with open(Forest.save_path, "rb") as f:
            self.roots = pickle.load(f)
    
    def __call__(self, figure):
        return Forest.infertrees(self.roots, Forest.discretize(figure.flatten()))


class SRModel:
    def __init__(self) -> None:
        with open(SR.save_path, "rb") as f:
            graph = pickle.load(f)
        self.graph = graph
        self.graph.eval()

    def __call__(self, figure):
        self.graph.flush()
        pred = self.graph.forward(figure, removelossnode=True)[-1]
        return np.argmax(pred, axis=-1)
    
class MLPModel:
    def __init__(self) -> None:
        with open(MLP.save_path, "rb") as f:
            graph = pickle.load(f)
        self.graph = graph
        self.graph.eval()

    def __call__(self, figure):
        self.graph.flush()
        pred = self.graph.forward(figure, removelossnode=True)[-1]
        return np.argmax(pred, axis=-1)

class MyModel:
    def __init__(self) -> None:
        with open("model/Q4_training_data.npy", "rb") as f:
            data = pickle.load(f)
        # Support both single model (Graph) and ensemble (list of Graphs)
        from autograd.BaseGraph import Graph as GraphCls
        if isinstance(data, GraphCls):
            self.graphs = [data]
        elif isinstance(data, list):
            self.graphs = data
        else:
            self.graphs = [data]
        for g in self.graphs:
            g.eval()

    def __call__(self, figure):
        # Auto-detect ConvNet vs MLP model and reshape input accordingly
        first_node = self.graphs[0][0]
        node_name = getattr(first_node, 'name', '')
        if 'conv2d' in node_name or 'inputnorm' in node_name:
            figure = figure.reshape(figure.shape[0], 1, 28, 28)

        preds = []
        for g in self.graphs:
            g.flush()
            pred = g.forward(figure, removelossnode=1)[-1]
            preds.append(pred)
        # Average log-probabilities then argmax
        avg_pred = np.mean(preds, axis=0)
        return np.argmax(avg_pred, axis=-1)



modeldict = {
    "Null": NullModel,
    "LR": LRModel,
    "Tree": TreeModel,
    "Forest": ForestModel,
    "SR": SRModel,
    "MLP": MLPModel,
    "Your": MyModel
}

