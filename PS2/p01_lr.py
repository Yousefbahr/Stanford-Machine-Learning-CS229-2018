# Important note: you do not have to modify this file for your homework.
import matplotlib.pyplot as plt
import util
import numpy as np


def J(theta, x, y):
    z = x.dot(theta)

    sigmoid_probs = 1. / (1 + np.exp(-z))

    return (1 / x.shape[0]) * np.sum(y * np.log(sigmoid_probs)
                                      + (1 - y) * np.log(1 - sigmoid_probs))

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
    learning_rate = 10
    thetas = []
    i = 1
    while True:
        print(learning_rate)
        i += 1
        prev_theta = theta
        grad = calc_grad(X, Y, theta)
        theta = theta - learning_rate * grad
        if i % 5000 == 0:

            print(f"J: {J(theta, X, Y)}")
            thetas.append(prev_theta)

        if i % 10000 == 0:
            print('Finished %d iterations' % i)

        if np.linalg.norm(prev_theta - theta) < 1e-15:
            print('Converged in %d iterations' % i)
            break

    return thetas

def logistic_regression_newton(x, y):
    theta = np.zeros(x.shape[1])
    iter = 0
    while True:
        iter += 1
        theta_old = np.copy(theta)

        print(f"J: {J(theta, x, y)}")

        # Sigmoid
        h_x = 1.0 / (1.0 + np.exp(-x.dot(theta)))

        # Hessian Matrix
        H = (x.T * h_x * (1 - h_x)).dot(x) / x.shape[0]

        # Gradients
        gradient_J_theta = x.T.dot(h_x - y) / x.shape[0]

        # Update theta
        theta -= np.linalg.inv(H).dot(gradient_J_theta)

        # change less than eps
        if np.linalg.norm(theta - theta_old, ord=1) < 1e-15:
            break


def main():
    print('==== Training model on data set A ====')
    Xa, Ya = util.load_csv('data/ds1_a.csv', add_intercept=True)

    # thetas_a = logistic_regression(Xa, Ya)


    print('\n==== Training model on data set B ====')
    Xb, Yb = util.load_csv('data/ds1_b.csv', add_intercept=True)


    # thetas_b = logistic_regression(Xb, Yb)
    # logistic_regression_newton(Xb, Yb)

    plt.figure(1)
    plt.subplot(221)
    # plt.scatter(Xa.T[1], Xa.T[2], c=Ya)
    plt.title("A dataset")

    plt.subplot(222)

    intervals = np.arange(5000, 30587, 5000)
    # objective_a = np.array([ J(theta, Xa, Ya) for theta in thetas_a])
    # plt.plot(intervals, objective_a)

    # plt.subplot(122)
    plt.subplot(223)
    plt.title("B dataset")

    # objective_b = np.array([J(theta, Xb, Yb) for theta in thetas_b])
    # plt.plot(intervals, objective_b)

    # plt.scatter(Xb.T[1], Xb.T[2], c=Yb)
    plt.show()


if __name__ == '__main__':
    main()
