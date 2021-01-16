import numpy as np
import util


def main(lr, train_path, eval_path, save_path, plot_path):
    """Problem: Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        save_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    # Run on the validation set, and use np.savetxt to save outputs to save_path
    clf = PoissonRegression(step_size=lr)
    clf.fit(x_train, y_train)

    # Evaluation
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    pred_eval = clf.predict(x_eval)
    pred_train = clf.predict(x_train)
    np.savetxt(save_path, pred_eval)

    # Plot
    plot_path_train = plot_path.replace(".png", "_train.png")
    plot_path_eval = plot_path.replace(".png", "_eval.png")
    util.plot_poisson(y_train, pred_train, plot_path_train)
    util.plot_poisson(y_eval, pred_eval, plot_path_eval)

    # *** END CODE HERE ***


class PoissonRegression:
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(
        self, step_size=1e-5, max_iter=10000000, eps=1e-5, theta_0=None, verbose=True
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
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        n_examples, dim = x.shape
        if self.theta is None:
            self.theta = np.zeros(dim)

        for iter in range(self.max_iter):
            pred = self.predict(x)
            l_dev = np.sum(x.T * (y - pred), axis=1) / n_examples
            theta_next = self.theta + self.step_size * l_dev
            if np.sum(np.abs(theta_next - self.theta)) < self.eps:
                print(iter)
                break
            self.theta = theta_next
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Floating-point prediction for each input, shape (n_examples,).
        """
        # *** START CODE HERE ***
        return np.exp(np.dot(x, self.theta))
        # *** END CODE HERE ***


if __name__ == "__main__":
    main(
        lr=1e-5,
        train_path="train.csv",
        eval_path="valid.csv",
        save_path="poisson_pred.txt",
        plot_path="poisson_plot.png",
    )

