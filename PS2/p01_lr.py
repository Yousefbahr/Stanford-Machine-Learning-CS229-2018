# Important note: you do not have to modify this file for your homework.
import matplotlib.pyplot as plt
import util
import numpy as np


def J(theta, x, y):
    return (-1 / x.shape[0]) * np.sum( np.log(1 / (1 + np.exp(-y * x.dot(theta)))))


def calc_grad(X, Y, theta):
    """Compute the gradient of the loss with respect to theta."""
    m, n = X.shape

    margins = Y * X.dot(theta)
    probs = 1. / (1 + np.exp(margins))
    grad = -(1./m) * (X.T.dot(probs * Y))

    return grad


def logistic_regression(X, Y):
    """Train a logistic regression model."""
    m, n = X.shape
    theta = np.zeros(n)
    # learning rate for dataset A
    learning_rate = 10

    # learning rate for dataset B
    # learning_rate = 1

    i = 1
    while True:
        i += 1
        prev_theta = theta
        grad = calc_grad(X, Y, theta)

        # add regularization to learn dataset B
        # theta = theta - learning_rate * (grad +  (10 / m) * np.r_[[0], theta[1:] ])

        # no need for regularization for dataset A
        theta = theta - learning_rate * grad

        # analyze loss
        if i % 5000 == 0:
            print(f"J: {J(theta, X, Y)}")

        if i % 10000 == 0:
            print('Finished %d iterations' % i)

        if np.linalg.norm(prev_theta - theta) < 1e-15:
            print('Converged in %d iterations' % i)
            break
    return


def main():
    print('==== Training model on data set A ====')
    Xa, Ya = util.load_csv('data/ds1_a.csv', add_intercept=True)
    logistic_regression(Xa, Ya)


    print('\n==== Training model on data set B ====')
    Xb, Yb = util.load_csv('data/ds1_b.csv', add_intercept=True)
    logistic_regression(Xb, Yb)

    # Plot the two datasets
    plt.figure(1)
    plt.subplot(121)
    plt.scatter(Xa.T[1], Xa.T[2], c=Ya)
    plt.title("A dataset")


    plt.subplot(122)
    plt.title("B dataset")
    plt.scatter(Xb.T[1], Xb.T[2], c=Yb)
    plt.show()


if __name__ == '__main__':
    main()
