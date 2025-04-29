import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


class Perceptron:
    """
    Multiclass perceptron with margin k.
    W: weight matrix of shape (n_classes, input_dim)
    """

    def __init__(self, input_dim, n_classes, lr=1.0, margin=0.0):
        self.W = np.zeros((n_classes, input_dim), dtype=np.float32)
        self.lr = lr
        self.margin = margin

    def fit(self, inputs, targets, epochs=10):
        """
        X: array, shape (n_samples, input_dim)
        y: array of ints in [0..n_classes-1]
        """
        for epoch in range(1, epochs + 1):
            idx = np.random.permutation(len(inputs))
            for i in idx:
                x = inputs[i]
                y = targets[i]
                scores = self.W.dot(x)  # shape (n_classes,)
                # if yi != y_pred:
                #     self.W[yi] += self.lr * xi
                #     self.W[y_pred] -= self.lr * xi
                # elif scores[yi] <= self.margin:
                #     self.W[yi] += self.lr * xi
                for idx in range(10):
                    state = 2 * (idx == y) - 1
                    if scores[idx] * state <= self.margin:
                        self.W[idx] += self.lr * state * x

    def predict(self, X):
        """
        X: array, shape (n_samples, input_dim)
        returns: array of preds shape (n_samples,)
        """
        scores = X.dot(self.W.T)  # shape (n_samples, n_classes)
        return np.argmax(scores, axis=1)


def main():
    # --- User-tunable parameters ---
    P = 10
    P_eval = 50
    N = 100  # projection dimension
    num_epochs = 100
    lr = 0.01
    margin = 5

    # 1) Load MNIST
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(int)

    # 2) Split into train (60 000) / test (10 000)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=60000, test_size=10000, random_state=42, shuffle=True
    )

    # 3) Subsample P*10 and P_eval*10 patterns
    rng = np.random.RandomState(42)
    train_idx = rng.choice(len(X_train), P * 10, replace=False)
    test_idx = rng.choice(len(X_test), P_eval * 10, replace=False)
    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_te, y_te = X_test[test_idx], y_test[test_idx]

    # 4) Random projection + binarize with sign()
    #    project to dimension N
    R = rng.randn(X.shape[1], N).astype(np.float32)
    X_tr_proj = np.sign(X_tr.dot(R))
    X_te_proj = np.sign(X_te.dot(R))
    # avoid zeros (np.sign(0) == 0) → map 0 → +1
    X_tr_proj[X_tr_proj == 0] = 1
    X_te_proj[X_te_proj == 0] = 1

    # train and evaluate perceptron after each epoch
    model = Perceptron(input_dim=N, n_classes=10, lr=lr, margin=margin)
    for epoch in range(num_epochs):
        model.fit(X_tr_proj, y_tr, epochs=1)

        if epoch % 1 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}")
            y_pred = model.predict(X_tr_proj)
            acc = np.mean(y_pred == y_tr) * 100
            print(f"Train accuracy on {len(y_tr)} examples: {acc:.2f}%")

            y_pred = model.predict(X_te_proj)
            acc = np.mean(y_pred == y_te) * 100
            print(f"Test accuracy on {len(y_te)} examples: {acc:.2f}%")


if __name__ == "__main__":
    main()
