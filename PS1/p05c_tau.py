import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Search tau_values for the best tau (lowest MSE on the validation set)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    x_test, y_test =  util.load_dataset(test_path, add_intercept=True)
    lowest_mse = float('inf')
    best_tau = None
    for tau in tau_values:
        model = LocallyWeightedLinearRegression(tau)
        model.fit(x_train, y_train)
        preds = model.predict(x_valid)
        plt.scatter(x_train.T[1], y_train)
        plt.scatter(x_valid.T[1], preds)
        plt.show()
        mse = np.mean((y_valid - preds) ** 2)
        if mse < lowest_mse:
            lowest_mse = mse
            best_tau = tau

    # Fit a LWR model with the best tau value
    model = LocallyWeightedLinearRegression(best_tau)
    model.fit(x_train, y_train)
    # Run on the test set to get the MSE value
    preds = model.predict(x_test)
    mse = np.mean((y_test - preds) ** 2)
    print(f'test set: tau={best_tau}, MSE={mse}')

    # Save predictions to pred_path
    np.save(pred_path, preds)
    # Plot data
    plt.scatter(x_test.T[1], y_test)
    plt.scatter(x_test.T[1], preds)
    plt.show()
    # *** END CODE HERE ***

