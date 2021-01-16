from typing import List
import numpy as np
import util
import sys

sys.path.append("../linearclass")

### NOTE : You need to complete logreg implementation first!

from logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/save_path
WILDCARD = "X"


def main(train_path, valid_path, test_path, save_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on t-labels,
        2. on y-labels,
        3. on y-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        save_path: Path to save predictions.
    """
    output_path_true = save_path.replace(WILDCARD, "true")
    output_path_naive = save_path.replace(WILDCARD, "naive")
    output_path_adjusted = save_path.replace(WILDCARD, "adjusted")

    plot_path_true = output_path_true.replace("pred.txt", "plot.png")
    plot_path_naive = output_path_naive.replace("pred.txt", "plot.png")
    plot_path_adjusted = output_path_adjusted.replace("pred.txt", "plot.png")

    # *** START CODE HERE ***
    # Part (a): Train and test on true labels
    # Make sure to save predicted probabilities to output_path_true using np.savetxt()

    # Load dataset (Ideal)
    x_train, t_train = util.load_dataset(train_path, label_col="t", add_intercept=True)
    x_valid, t_valid = util.load_dataset(valid_path, label_col="t", add_intercept=True)
    x_test, t_test = util.load_dataset(test_path, label_col="t", add_intercept=True)

    # Load classifier
    clf = LogisticRegression()

    # Fit classifier with training set
    clf.fit(x_train, t_train)

    # Prediction
    pred_train = clf.predict(x_train)
    pred_valid = clf.predict(x_valid)
    pred_test = clf.predict(x_test)

    # Output txt and plot.png files
    np.savetxt(output_path_true, pred_test)
    print("Ideal case (a)")
    print(f"ACC_Train : {np.sum((pred_train > 0.5) == t_train) / len(t_train):.4f}")
    print(f"ACC_Valid : {np.sum((pred_valid > 0.5) == t_valid) / len(t_valid):.4f}")
    print(f"ACC_Test  : {np.sum((pred_test > 0.5) == t_test) / len(t_test):.4f}")
    util.plot(
        x_train,
        t_train,
        clf.theta,
        plot_path_true.replace(".png", "_train.png"),
        correction=1.0,
    )
    util.plot(
        x_valid,
        t_valid,
        clf.theta,
        plot_path_true.replace(".png", "_valid.png"),
        correction=1.0,
    )
    util.plot(
        x_test,
        t_test,
        clf.theta,
        plot_path_true.replace(".png", "_test.png"),
        correction=1.0,
    )

    # Part (b): Train on y-labels and test on true labels
    # Make sure to save predicted probabilities to output_path_naive using np.savetxt()

    # Load dataset (Ideal)
    x_train, y_train = util.load_dataset(train_path, label_col="y", add_intercept=True)
    x_valid, t_valid = util.load_dataset(valid_path, label_col="t", add_intercept=True)
    x_test, t_test = util.load_dataset(test_path, label_col="t", add_intercept=True)

    # Load classifier
    clf = LogisticRegression()

    # Fit classifier with training set
    clf.fit(x_train, y_train)

    # Prediction
    pred_train = clf.predict(x_train)
    pred_valid = clf.predict(x_valid)
    pred_test = clf.predict(x_test)

    # Output txt and plot.png files
    np.savetxt(output_path_naive, pred_test)
    print("Naive case (b)")
    print(f"ACC_Train : {np.sum((pred_train > 0.5) == y_train) / len(y_train):.4f}")
    print(f"ACC_Valid : {np.sum((pred_valid > 0.5) == t_valid) / len(t_valid):.4f}")
    print(f"ACC_Test  : {np.sum((pred_test > 0.5) == t_test) / len(t_test):.4f}")
    util.plot(
        x_train,
        y_train,
        clf.theta,
        plot_path_naive.replace(".png", "_train.png"),
        correction=1.0,
    )
    util.plot(
        x_valid,
        t_valid,
        clf.theta,
        plot_path_naive.replace(".png", "_valid.png"),
        correction=1.0,
    )
    util.plot(
        x_test,
        t_test,
        clf.theta,
        plot_path_naive.replace(".png", "_test.png"),
        correction=1.0,
    )

    # Part (f): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to output_path_adjusted

    # Load dataset (Ideal)
    x_valid, y_valid = util.load_dataset(valid_path, label_col="y", add_intercept=True)
    x_valid, t_valid = util.load_dataset(valid_path, label_col="t", add_intercept=True)
    x_test, t_test = util.load_dataset(test_path, label_col="t", add_intercept=True)

    # Calculate h(x) = p(y=1|x)
    clf_h = LogisticRegression()
    clf_h.fit(x_valid, y_valid)
    y_pred_valid = clf_h.predict(x_valid)
    alpha = (y_pred_valid * y_valid).sum() / y_valid.sum()

    # Re train model w. predicted t_valid
    predicted_t_valid = y_pred_valid / alpha > 0.5
    clf = LogisticRegression()
    clf.fit(x_valid, predicted_t_valid)

    # Predict valid / test
    pred_valid = clf.predict(x_valid)
    pred_test = clf.predict(x_test)
    print("Adjusted case (f)")
    print(f"ACC_Valid : {np.sum((pred_valid > 0.5) == t_valid) / len(t_valid):.4f}")
    print(f"ACC_Test  : {np.sum((pred_test > 0.5) == t_test) / len(t_test):.4f}")

    np.savetxt(output_path_adjusted, pred_test)
    util.plot(
        x_valid,
        t_valid,
        clf.theta,
        plot_path_adjusted.replace(".png", "_valid.png"),
        correction=1.0,
    )
    util.plot(
        x_test,
        t_test,
        clf.theta,
        plot_path_adjusted.replace(".png", "_test.png"),
        correction=1.0,
    )

    # *** END CODER HERE


if __name__ == "__main__":
    main(
        train_path="train.csv",
        valid_path="valid.csv",
        test_path="test.csv",
        save_path="posonly_X_pred.txt",
    )

