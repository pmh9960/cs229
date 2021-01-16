import util
import numpy as np
import matplotlib.pyplot as plt

np.seterr(all="raise")


factor = 2.0


class LinearModel(object):
    """Base class for linear models."""

    def __init__(self, theta=None):
        """
        Args:
            theta: Weights vector for the model.
        """
        self.theta = theta

    def fit(self, X, y):
        """Run solver to fit linear model. You have to update the value of
        self.theta using the normal equations.

        Args:
            X: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        # Normal equation : X^T X theta = X^T y
        self.theta = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y))
        # *** END CODE HERE ***

    def create_poly(self, k, X):
        """
        Generates a polynomial feature map using the data x.
        The polynomial map should have powers from 0 to k
        Output should be a numpy array whose shape is (n_examples, k+1)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        for i in range(k - 1):
            x_col = X.T[1] ** (i + 2)
            X = np.vstack([X.T, x_col]).T
        return X
        # *** END CODE HERE ***

    def create_sin(self, k, X):
        """
        Generates a sin with polynomial featuremap to the data x.
        Output should be a numpy array whose shape is (n_examples, k+2)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        X = self.create_poly(k, X)
        x_col = np.sin(X.T[1])
        X = np.vstack([X.T, x_col]).T
        return X
        # *** END CODE HERE ***

    def predict(self, X):
        """
        Make a prediction given new inputs x.
        Returns the numpy array of the predictions.

        Args:
            X: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return np.dot(X, self.theta)
        # *** END CODE HERE ***


def run_exp(
    train_path,
    valid_path,
    sine=False,
    valid=False,
    ks=[1, 2, 3, 5, 10, 20],
    filename="plot_DATA_REG.png",
):
    train_x, train_y = util.load_dataset(train_path, add_intercept=True)
    valid_x, valid_y = util.load_dataset(valid_path, add_intercept=True)
    plot_x = np.ones([1000, 2])
    plot_x[:, 1] = np.linspace(-factor * np.pi, factor * np.pi, 1000)
    plt.figure()
    if valid:
        plt.scatter(valid_x[:, 1], valid_y)
        filename = filename.replace("DATA", "valid")
    else:
        plt.scatter(train_x[:, 1], train_y)
        filename = filename.replace("DATA", "train")

    if sine:
        filename = filename.replace("REG", "sine")
    else:
        filename = filename.replace("REG", "poly")

    for k in ks:
        """
        Our objective is to train models and perform predictions on plot_x data
        """
        # *** START CODE HERE ***
        # Train
        reg = LinearModel()
        X = reg.create_poly(k, train_x)
        if sine:
            X = reg.create_sin(k, train_x)
        reg.fit(X, train_y)

        # Plot
        plot_X = reg.create_poly(k, plot_x)
        if sine:
            plot_X = reg.create_sin(k, plot_x)
        plot_y = reg.predict(plot_X)

        # *** END CODE HERE ***
        """
        Here plot_y are the predictions of the linear model on the plot_x data
        """
        plt.ylim(-2, 2)
        plt.plot(plot_x[:, 1], plot_y, label="k=%d" % k)

    plt.legend()
    plt.savefig(filename)
    plt.clf()


def main(train_path, small_path, eval_path, plot_path):
    """
    Run all expetriments
    """
    # *** START CODE HERE ***
    run_exp(train_path, eval_path, sine=False)
    run_exp(train_path, eval_path, sine=False, valid=True)
    run_exp(train_path, eval_path, sine=True)
    run_exp(train_path, eval_path, sine=True, valid=True)
    run_exp(small_path, eval_path, filename="plot_small_DATA_REG.png")
    run_exp(small_path, eval_path, valid=True, filename="plot_small_DATA_REG.png")
    run_exp(small_path, eval_path, sine=True, filename="plot_small_DATA_REG.png")
    run_exp(
        small_path,
        eval_path,
        sine=True,
        valid=True,
        filename="plot_small_DATA_REG.png",
    )
    # *** END CODE HERE ***


if __name__ == "__main__":
    main(
        train_path="train.csv",
        small_path="small.csv",
        eval_path="test.csv",
        plot_path="plot.png",
    )

