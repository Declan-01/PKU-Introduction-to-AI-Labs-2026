
import mnist, numpy as np, pickle, time
from util import setseed
from autograd.utils import PermIterator
from autograd.BaseGraph import Graph
from autograd.BaseNode import *
from scipy import ndimage

setseed(0)
save_path = "model/Q4_training_data.npy"

lr = 1.5e-3; wd2 = 2e-5; batchsize = 256; epochs = 9; aug_ratio = 0.25
begin_time = time.time()

def augment_img(x):
    """Shift-heavy augmentation."""
    img = x.reshape(28, 28)
    r = np.random.random()
    if r < 0.55:
        dx = np.random.randint(-6, 7); dy = np.random.randint(-6, 7)
        img = ndimage.shift(img, [dy, dx], order=1)
    elif r < 0.75:
        angle = np.random.uniform(-18, 18)
        img = ndimage.rotate(img, angle, reshape=False, order=1)
    elif r < 0.90:
        angle = np.random.uniform(-12, 12)
        img = ndimage.rotate(img, angle, reshape=False, order=1)
        dx = np.random.randint(-3, 4); dy = np.random.randint(-3, 4)
        img = ndimage.shift(img, [dy, dx], order=1)
    elif r < 0.97:
        scale = np.random.uniform(0.85, 1.15)
        img = ndimage.zoom(img, zoom=scale, order=1)
        h, w = img.shape
        if h > 28: img = img[(h-28)//2:(h-28)//2+28, :]
        if w > 28: img = img[:, (w-28)//2:(w-28)//2+28]
        h, w = img.shape
        if h < 28 or w < 28:
            ph, pw = max(0,28-h), max(0,28-w)
            img = np.pad(img, ((ph//2,ph-ph//2),(pw//2,pw-pw//2)))
    return img

if __name__ == "__main__":
    X_flat = np.concatenate([mnist.trn_X, mnist.val_X], axis=0)
    Y = np.concatenate([mnist.trn_Y, mnist.val_Y], axis=0)
    N = X_flat.shape[0]

    print(f"Q4 ConvNet v36: {N} samples, lr={lr}, wd2={wd2}, bs={batchsize}, ep={epochs}, aug={aug_ratio}")

    graph = Graph([
        InputNorm(),
        Conv2d(1, 8, 3, padding=1), relu(),
        MaxPool2d(2),
        Flatten(),
        Linear(8 * 14 * 14, 256), relu(), Dropout(0.2),
        Linear(256, 128), relu(),
        Linear(128, 10), LogSoftmax(), NLLLoss(Y),
    ])

    dataloader = PermIterator(N, batchsize)
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        t_haty, t_y, t_loss = [], [], []
        graph.train()

        for perm in dataloader:
            t_X = X_flat[perm].reshape(-1, 1, 28, 28)
            t_Y = Y[perm]

            n_aug = max(1, int(len(perm) * aug_ratio))
            t_X_aug = np.array([augment_img(X_flat[perm[i]]) for i in range(n_aug)])
            t_X_aug = t_X_aug.reshape(-1, 1, 28, 28)

            t_Xc = np.concatenate([t_X, t_X_aug], axis=0)
            t_Yc = np.concatenate([t_Y, t_Y[:n_aug]])

            graph[-1].y = t_Yc; graph.flush()
            pred, loss = graph.forward(t_Xc)[-2:]
            t_haty.append(np.argmax(pred, axis=1)); t_y.append(t_Yc)
            graph.backward(); graph.optimstep(lr, 0, wd2); t_loss.append(loss)

        train_acc = np.average(np.concatenate(t_haty) == np.concatenate(t_y))
        et = time.time() - t0
        print(f"ep{epoch:2d} loss{np.average(t_loss):.3e} tr{train_acc:.4f} t{et:.0f}s")

    # Save final model (epoch 9 = peak, no selection needed)
    graph.eval(); graph.flush()
    with open(save_path, "wb") as f:
        pickle.dump(graph, f)

    print(f"\nTrain time: {time.time()-t0:.0f}s  Total: {time.time()-begin_time:.0f}s")
