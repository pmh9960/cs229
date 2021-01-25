import numpy as np
import util


def main(train_path, valid_path, save_path, plot_path):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Train a GDA classifier
    # Plot decision boundary on validation set
    # Use np.savetxt to save outputs from validation set to save_path
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=False)

    clf = GDA()
    clf.fit(x_train, y_train)
    pred_valid = clf.predict(x_valid)
    np.savetxt(save_path, pred_valid)
    print(f"ACC_valid : {np.sum((pred_valid > 0.5) == y_valid) / len(y_valid)}")
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


class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(
        self, step_size=0.01, max_iter=10000, eps=1e-5, theta_0=None, verbose=True
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
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        # Find phi, mu_0, mu_1, and sigma
        # Write theta in terms of the parameters
        n, d = x.shape
        if self.theta is None:
            self.theta = np.zeros((d,))

        phi = np.mean(y)
        mu_0 = (np.sum((1 - y) * x.T, axis=1) / np.sum(1 - y)).T
        mu_1 = (np.sum((y) * x.T, axis=1) / np.sum(y)).T
        # print(mu_0, mu_1)
        mu_y = (1 - y).reshape(n, 1) * mu_0 + y.reshape(n, 1) * mu_1
        sigma = 0
        for i in range(n):
            sigma += np.dot((x - mu_y)[i].reshape(d, 1), (x - mu_y)[i].reshape(d, 1).T)
        sigma /= n
        assert mu_0.shape == (d,)
        assert mu_1.shape == (d,)
        assert mu_y.shape == (n, d)
        assert sigma.shape == (d, d)
        sigma_inv = np.linalg.inv(sigma)

        theta = np.dot(sigma_inv, mu_1 - mu_0)
        theta_0 = (
            np.dot(mu_0.T, np.dot(sigma_inv, mu_0)) / 2
            - np.dot(mu_1.T, np.dot(sigma_inv, mu_1)) / 2
            - np.log((1 - phi) / phi)
        )
        self.theta = np.append(theta_0, theta)
        assert self.theta.shape == (d + 1,)
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return 1 / (1 + np.exp(-(np.dot(self.theta[1:], x.T) + self.theta[0])))
        # *** END CODE HERE


if __name__ == "__main__":
    main(
        train_path="ds1_train_log.csv",
        valid_path="ds1_valid_log.csv",
        save_path="gda_pred_1_log.txt",
        plot_path="gda_plot_1_log.png",
    )
    main(
        train_path="ds2_train.csv",
        valid_path="ds2_valid.csv",
        save_path="gda_pred_2.txt",
        plot_path="gda_plot_2.png",
    )

