import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    x_val, y_val = util.load_dataset(eval_path, add_intercept=True)
    # Fit a LWR model
    model = LocallyWeightedLinearRegression(tau)
    model.fit(x_train, y_train)
    preds = model.predict(x_val)
    # Get MSE value on the validation set
    mse = np.mean((y_val - preds) ** 2)
    # Plot validation predictions on top of training set

    plt.scatter(x_train.T[1], y_train)
    plt.scatter(x_val.T[1], preds)
    plt.show()
    # No need to save predictions
    # Plot data
    # *** END CODE HERE ***


def MSE(true, predicted):
    assert true.shape[0] == predicted.shape[0]
    return (1 / true.shape[0]) * (true - predicted) ** 2

class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.x = x
        self.y = y
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        m = x.shape[0]
        targets = np.zeros((m,))

        for i in range(m):
            w = np.diag(np.exp(-np.sum((self.x - x[i]) ** 2, axis=1) / (2 * self.tau ** 2)))
            XT_W = self.x.T @ w
            theta = np.linalg.inv(XT_W @ self.x) @ XT_W @ self.y
            targets[i] =  x[i] @ theta

        return targets
        # *** END CODE HERE ***

if __name__ == "__main__":
    main(0.5, "data/ds5_train.csv", "data/ds5_valid.csv")

