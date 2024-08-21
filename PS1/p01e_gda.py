import numpy
import numpy as np
import util
import matplotlib.pyplot as plt
from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    x_valid, y_valid = util.load_dataset(eval_path, add_intercept=False)
    clf = GDA()
    clf.fit(x_train, y_train)
    preds = clf.predict(x_valid).tolist()
    np.savetxt(pred_path, preds)


    # plt.subplots(1, 2)
    #
    # plt.subplot(1, 2 ,1)
    # plt.scatter(x_train.T[0], x_train.T[1], marker='o' , c=y_train)
    # plt.title(f"GDA True Labels\n Accuracy: {accuracy}")
    #
    # plt.subplot(1, 2, 2)
    # plt.scatter(x_train.T[0], x_train.T[1], marker='o', c=preds)
    # plt.title("GDA Preds")

    # plt.show()
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
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

    def phi(self, x, y):
        return (1 / x.shape[0]) * (np.sum(y == 1))

    def mu0(self, x, y):
        return (y == 0) @ x / np.sum(y == 0)

    def mu1(self, x, y):
        return (y == 1) @ x / np.sum(y == 1)

    def covariance_mat(self, x, y):
        mu0 = self.mu0(x, y)
        q0 = (x - mu0)
        outer_products = np.array([np.outer(d, d) for d in q0])
        return np.sum(outer_products, axis=0) / x.shape[0]

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        phi = self.phi(x, y)
        mu0 = self.mu0(x, y)
        mu1 = self.mu1(x, y)
        covar_mat = self.covariance_mat(x, y)
        self.theta = [phi, mu0, mu1, covar_mat]
        return self.theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        phi, mu0, mu1, covar_mat = self.theta
        det = np.linalg.det(covar_mat)
        inv = np.linalg.inv(covar_mat)
        constant = (1 / (2 * np.pi) ** (n/2) * det ** (1/2))

        # mu for class 0 and class 1
        mu = [mu0, mu1]
        # phi for class 0 and class 1
        phis = [1 - phi, phi]
        # all predictions, a column for class 0 and column for class 1
        my_preds = np.zeros((x.shape[0], 2))
        for i in range(2):
            probab = constant * np.exp(-0.5 * np.sum((x - mu[i]) @ inv * (x - mu[i]), axis=1)) * phis[i]
            my_preds.T[i] = probab

        return np.argmax(my_preds, axis=1)
        # *** END CODE HERE


# if __name__ == "__main__":
#     main("data/ds1_train.csv", "data/ds1_valid.csv","")
#     main("data/ds2_train.csv", "data/ds2_valid.csv","")
