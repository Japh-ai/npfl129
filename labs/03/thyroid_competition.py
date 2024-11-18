#!/usr/bin/env python3
# Jorge Antonio Puente Huerta
import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request

import numpy as np
import numpy.typing as npt
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import PolynomialFeatures

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument(
    "--predict", default=None, type=str, help="Path to the dataset to predict"
)
parser.add_argument(
    "--recodex", default=False, action="store_true", help="Running in ReCodEx"
)
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument(
    "--model_path", default="thyroid_competition.model", type=str, help="Model path"
)


class Dataset:
    """Thyroid Dataset.

    The dataset contains real medical data related to thyroid gland function,
    classified either as normal or irregular (i.e., some thyroid disease).
    The data consists of the following features in this order:
    - 15 binary features
    - 6 real-valued features

    The target variable is binary, with 1 denoting a thyroid disease and
    0 normal function.
    """

    def __init__(
        self,
        name="thyroid_competition.train.npz",
        url="https://ufal.mff.cuni.cz/~courses/npfl129/2425/datasets/",
    ):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        # Load the dataset and return the data and targets.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value)


def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        X_train = train.data  # Shape: (num_samples, 21)
        y_train = train.target  # Shape: (num_samples,)

        # Define preprocessing for different feature types
        preprocessor = ColumnTransformer(
            transformers=[
                # Scale continuous features
                ("continuous", StandardScaler(), list(range(15, 21))),
                ("poly", PolynomialFeatures(degree=2), list(range(21))),
            ]
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    LogisticRegression(
                        random_state=args.seed,
                        max_iter=1000,  # Increase iterations to ensure convergence
                        solver="liblinear",  # Solver that supports both l1 and l2 penalties
                    ),
                ),
            ]
        )

        # Define the hyperparameter grid for GridSearchCV
        param_grid = {
            "classifier__C": [0.01, 0.1, 1, 10, 100],  # Regularization strength
            "classifier__penalty": ["l1", "l2"],  # Regularization type
            # Note: 'liblinear' solver supports both 'l1' and 'l2' penalties
        }

        # Define the cross-validation strategy
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

        # Initialize GridSearchCV
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="accuracy",  # Evaluation metric
            cv=cv,
            n_jobs=-1,  # Utilize all available CPU cores
            verbose=1,  # Set to 1 to see progress; set to 0 to disable
        )

        # TODO: Train a model on the given dataset and store it in `model`.
        # Fit GridSearchCV on the training data
        grid_search.fit(X_train, y_train)

        # Retrieve the best model from GridSearchCV
        best_model = grid_search.best_estimator_

        # Optionally, print out the best parameters and best score
        print("Best Parameters:", grid_search.best_params_)
        print(
            "Best Cross-Validation Accuracy: {:.2f}%".format(
                100 * grid_search.best_score_
            )
        )
        model = best_model

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions, either as a Python list or a NumPy array.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions.
        predictions = model.predict(test.data)

        return predictions


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
