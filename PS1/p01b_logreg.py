import numpy as np
import util
import matplotlib.pyplot as plt
from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    x_valid, y_valid = util.load_dataset(eval_path, add_intercept=True)
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    preds = np.round(clf.predict(x_valid).tolist())
    np.savetxt(pred_path, preds)
    accuracy = clf.accuracy(y_valid, preds)

    # util.plot(x_train, y_train, clf.theta)
    # plt.title(f"Logistic Regression\n Accuracy: {accuracy}")
    # plt.show()
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def accuracy(self, true, preds):
        y_true = np.array(true)
        y_pred = np.array(preds)

        # Calculate the number of correct predictions
        correct_predictions = np.sum(y_true == y_pred)

        # Calculate the total number of samples
        total_samples = len(y_true)

        # Calculate accuracy
        accuracy = correct_predictions / total_samples

        return accuracy

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x.dot(self.theta)))

    def J(self, x, y):
        # Binary Cross Entropy loss (Log loss)
        sigmoid_probs = self.sigmoid(x)
        return (-1 / x.shape[0]) * np.sum(y * np.log(sigmoid_probs)
                      + (1 - y) * np.log(1 - sigmoid_probs))

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

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
            print(f"Loss Before Training: {self.J(x, y)}")

        while iter <= self.max_iter:
            theta_old = np.copy(self.theta)

            # Sigmoid
            h_x = self.sigmoid(x)

            # Hessian Matrix
            H = (x.T * h_x * (1 - h_x)).dot(x) / x.shape[0]

            # Gradients
            gradient_J_theta = x.T.dot(h_x - y) / x.shape[0]

            # Update theta
            self.theta -= np.linalg.inv(H).dot(gradient_J_theta)

            loss = self.J(x, y)

            # change less than eps
            if np.linalg.norm(self.theta - theta_old, ord=1) < self.eps:
                break

            if self.verbose:
                print(f"Iteration: {iter} | Loss: {loss}")

            iter += 1
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return self.sigmoid(x)
        # *** END CODE HERE ***

# if __name__ == "__main__":
#     main("data/ds1_train.csv", "data/ds1_valid.csv","")
#     main("data/ds2_train.csv", "data/ds2_valid.csv","")
#     # main("mine.csv", "data/ds1_valid.csv","")