#!/usr/bin/env python3
# Jorge Antonio Puente Huerta
import argparse

import numpy as np
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument(
    "--plot",
    default=False,
    const=True,
    nargs="?",
    type=str,
    help="Plot the predictions",
)
parser.add_argument(
    "--recodex", default=False, action="store_true", help="Running in ReCodEx"
)
parser.add_argument("--seed", default=13, type=int, help="Random seed")
parser.add_argument(
    "--test_size",
    default=0.5,
    type=lambda x: int(x) if x.isdigit() else float(x),
    help="Test size",
)
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[float, float]:
    # Load the diabetes dataset.
    dataset = sklearn.datasets.load_diabetes()

    # TODO: Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target = (
        sklearn.model_selection.train_test_split(
            dataset.data,
            dataset.target,
            test_size=args.test_size,
            random_state=args.seed,
        )
    )
    lambdas = np.geomspace(0.01, 10, num=500)
    # TODO: Using `sklearn.linear_model.Ridge`, fit the train set using
    # L2 regularization, employing the above defined lambdas.
    # For every model, compute the root mean squared error and return the
    # lambda producing lowest RMSE and the corresponding RMSE.

    best_lambda = None
    best_rmse = float("inf")
    rmses = []
    for lam in lambdas:
        # Initialize the Ridge Regression model with the current lambda (alpha).
        model = sklearn.linear_model.Ridge(alpha=lam, random_state=args.seed)

        # Fit the model on the training data.
        model.fit(train_data, train_target)

        # Predict on the test data.
        predictions = model.predict(test_data)

        # Compute RMSE
        rmse = sklearn.metrics.root_mean_squared_error(test_target, predictions)

        # Append the RMSE to the list for plotting purposes.
        rmses.append(rmse)

        # Update the best lambda and best RMSE if the current RMSE is lower.
        if rmse < best_rmse:
            best_rmse = rmse
            best_lambda = lam

    if args.plot:
        # This block is not required to pass in ReCodEx; however, it is useful
        # to learn to visualize the results. If you collect the respective
        # results for `lambdas` to an array called `rmses`, the following lines
        # will plot the result if you add `--plot` argument.
        import matplotlib.pyplot as plt

        plt.plot(lambdas, rmses)
        plt.xscale("log")
        plt.xlabel("L2 regularization strength $\\lambda$")
        plt.ylabel("RMSE")
        (
            plt.show()
            if args.plot is True
            else plt.savefig(args.plot, transparent=True, bbox_inches="tight")
        )

    return best_lambda, best_rmse


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    best_lambda, best_rmse = main(main_args)
    print("{:.2f} {:.2f}".format(best_lambda, best_rmse))
