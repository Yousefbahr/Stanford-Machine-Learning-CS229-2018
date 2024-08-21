import numpy as np
import util
import matplotlib.pyplot as plt
from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    # Part (c): Train and test on true labels
    """
    Train on x_train, t_train, where t is the true labels
    Test on x_test, t_test  --> should perform well
    """
    x_train, t_train = util.load_dataset(train_path, label_col='t' , add_intercept=True)
    x_test, t_test = util.load_dataset(test_path, label_col='t' , add_intercept=True)
    lg_e = LogisticRegression(verbose=False)
    lg_e.fit(x_train, t_train)
    t_preds = np.round(lg_e.predict(x_test))
    np.savetxt(pred_path_c, t_preds)
    util.plot(x_test, t_test, lg_e.theta)
    plt.show()
    print(lg_e.accuracy(t_test, t_preds))
    # Make sure to save outputs to pred_path_c


    # Part (d): Train on y-labels and test on true labels
    """
    Train on x_train, y_train , where y = 1 is the positive labels only, y = 0 can be positive or negative label
    Test on x_test, y_test --> should perform poorly
    """
    x_train, y_train = util.load_dataset(train_path,  add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, label_col='y', add_intercept=True)

    lg_d = LogisticRegression(verbose=False)
    lg_d.fit(x_train, y_train)
    y_preds = np.round(lg_d.predict(x_test))
    np.savetxt(pred_path_d, y_preds)
    util.plot(x_test, y_test, lg_d.theta)
    plt.show()
    print(lg_d.accuracy(y_test, y_preds))
    # Make sure to save outputs to pred_path_d


    # Part (e): Apply correction factor using validation set and test on true labels
    """
    Trained on x_train, y_train
    Test on x_train, t_train with alpha (calculated from validation dataset) to correct the predictions 
    --> perform as nearly well as training on t_train 
    """
    x_val, y_val = util.load_dataset(valid_path, add_intercept=True)

    def h(theta, x):
        return 1 / (1 + np.exp(-np.dot(x, theta)))

    v_plus = x_val[y_val == 1]
    alpha = h(lg_d.theta, v_plus).mean()

    theta_prime = lg_d.theta + np.log(2 / alpha - 1) * np.array([1, 0, 0])

    t_preds_e = lg_d.predict(x_test) / alpha >= 0.5

    util.plot(x_test, y_test, theta_prime)
    plt.show()
    print(lg_d.accuracy(t_test, t_preds_e))
    np.savetxt(pred_path_e, t_preds_e)
    # Plot and use np.savetxt to save outputs to pred_path_e
    # *** END CODER HERE


# if __name__ == "__main__":
#     main("data/ds3_train.csv", "data/ds3_valid.csv", "data/ds3_test.csv", "")