import numpy as np
import util
import matplotlib.pyplot as plt
from linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    # Run on the validation set, and use np.savetxt to save outputs to pred_path
    x_val, y_val = util.load_dataset(eval_path, add_intercept=False)
    model = PoissonRegression(step_size=lr, verbose=False)
    model.fit(x_train, y_train)
    preds = model.predict(x_val)
    np.savetxt(pred_path, preds)
    # *** END CODE HERE ***


class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def poisson_loss(self, x, y):
        lambda_ = np.exp(np.dot(x, self.theta))

        loss = np.sum(lambda_ - y * np.dot(x, self.theta))
        return loss

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        # no specified value for theta_0
        if not self.theta:
            self.theta = np.zeros((x.shape[1]))
        # specified value for theta_0
        else:
            self.theta = np.concatenate(([self.theta], np.zeros(x.shape[1] - 1)))

        iter = 0
        if self.verbose:
            print(f"Loss Before Training: {self.poisson_loss(x, y):.5e}")

        while iter <= self.max_iter:
            theta_old = np.copy(self.theta)

            exponential = np.exp(x.dot(self.theta))

            self.theta += self.step_size / x.shape[0] * (y - exponential).dot(x)

            # change less than eps
            if np.linalg.norm(self.theta - theta_old, ord=1) < self.eps:
                break

            if self.verbose:
                print(f"Iter: {iter} | Loss: {self.poisson_loss(x, y):.5e}")
            iter += 1
        # **
        # * END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***
        return np.exp(x.dot(self.theta))
        # *** END CODE HERE ***

# if __name__ == "__main__":
#     main(1e-7, "data/ds4_train.csv", "data/ds4_valid.csv","")