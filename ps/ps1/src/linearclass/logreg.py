import numpy as np
import util

import os


def main(train_path, valid_path, save_path, plot_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Train a logistic regression classifier
    # Plot decision boundary on top of validation set set
    # Use np.savetxt to save predictions on eval set to save_path
    clf = LogisticRegression()
    clf.fit(x_train, y_train)

    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    pred = clf.predict(x_valid)
    np.savetxt(save_path, pred)
    print(f"ACC_valid : {np.sum((pred > 0.5) == y_valid) / len(y_valid)}")
    util.plot(
        x_train,
        y_train,
        clf.theta,
        plot_path.replace(".png", "_train.png"),
        correction=1.0,
    )
    util.plot(
        x_valid,
        y_valid,
        clf.theta,
        plot_path.replace(".png", "_valid.png"),
        correction=1.0,
    )
    # *** END CODE HERE ***


class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(
        self, step_size=0.01, max_iter=1000000, eps=1e-5, theta_0=None, verbose=True
    ):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        n_examples, dim = x.shape
        if self.theta is None:
            self.theta = np.zeros(dim)
        for iter in range(self.max_iter):
            # Prediction value
            pred = self.predict(x)
            # Loss function
            # l = -np.sum(y * np.log(pred) + (1 - y) * np.log(1 - pred)) / n_examples
            l_dev = -1 / n_examples * np.sum(x.T * (y - pred), axis=1)
            # Hessian matrix
            H = np.zeros((dim, dim))
            for i in range(dim):
                for j in range(dim):
                    H[i][j] = np.sum(pred * (1 - pred) * x.T[i] * x.T[j]) / n_examples
            # Inverse of Hessian matrix
            try:
                H_inv = np.linalg.inv(H)
            except:
                break
            theta_next = self.theta - np.dot(H_inv, l_dev)
            if np.sum(np.abs(theta_next - self.theta)) < self.eps:
                break
            self.theta = theta_next

        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return 1 / (1 + np.exp(-np.dot(x, self.theta)))
        # *** END CODE HERE ***


if __name__ == "__main__":
    rel_path = os.path.join("ps", "ps1", "src", "linearclass")
    main(
        train_path="ds1_train.csv",
        valid_path="ds1_valid.csv",
        save_path="logreg_pred_1.txt",
        plot_path="logreg_plot_1.png",
    )
    main(
        train_path="ds2_train.csv",
        valid_path="ds2_valid.csv",
        save_path="logreg_pred_2.txt",
        plot_path="logreg_plot_2.png",
    )

