import numpy as np
import csv


def import_data():
    train_x = np.genfromtxt("train_X_lr.csv", dtype=np.float64, delimiter=',', skip_header=1)
    train_y = np.genfromtxt("train_Y_lr.csv", dtype=np.float64, delimiter=',')
    return train_x, train_y


def compute_cost(X, Y, W):
    Y_pred = np.dot(X, W)
    mse = np.sum(np.square(Y_pred - Y))
    cost_value = mse / (2 * len(X))
    return cost_value


def compute_gradient_of_cost_function(X, Y, W):
    Y_pred = np.dot(X, W)
    difference = Y_pred - Y
    dW = (1 / len(X)) * (np.dot(difference.T, X))
    dW = dW.T
    return dW


def optimize_weights_using_gradient_descent(X, Y, W, num_iterations, learning_rate):
    iter = 0
    prev_iter_cost = 0

    while True:
        iter = iter + 1
        dW = compute_gradient_of_cost_function(X, Y, W)
        W = W - learning_rate * dW
        cost = compute_cost(X, Y, W)
        if abs(prev_iter_cost - cost) < 0.000001:
            print(iter, cost)
            break
        prev_iter_cost = cost

    return W


def train_model(X, Y):
    X = np.insert(X, 0, 1, axis=1)
    Y = Y.reshape(len(X), 1)
    W = np.zeros((X.shape[1], 1))
    W = optimize_weights_using_gradient_descent(X, Y, W, 10, 0.0002)
    return W


def save_model(weight, weights_file_name):
    with open(weights_file_name, 'w') as weights_file:
        wr = csv.writer(weights_file)
        wr.writerows(weights)
        weights_file.close()


if __name__ == "__main__":
    X, Y = import_data()
    weights = train_model(X, Y)
    save_model(weights, "WEIGHTS_FILE.csv")
