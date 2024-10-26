#!/usr/bin/env python3
# Jorge Antonio Puente Huerta
import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request
import pandas as pd
import numpy as np
import numpy.typing as npt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error


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
    "--model_path", default="rental_competition.model", type=str, help="Model path"
)


class Dataset:
    """Rental Dataset.

    The dataset instances consist of the following 12 features:
    - season (1: winter, 2: spring, 3: summer, 4: autumn)
    - year (0: 2011, 1: 2012)
    - month (1-12)
    - hour (0-23)
    - holiday (binary indicator)
    - day of week (0: Sun, 1: Mon, ..., 6: Sat)
    - working day (binary indicator; a day is neither weekend nor holiday)
    - weather (1: clear, 2: mist, 3: light rain, 4: heavy rain)
    - temperature (normalized so that -8 Celsius is 0 and 39 Celsius is 1)
    - feeling temperature (normalized so that -16 Celsius is 0 and 50 Celsius is 1)
    - relative humidity (0-1 range)
    - windspeed (normalized to 0-1 range)

    The target variable is the number of rented bikes in the given hour.
    """

    def __init__(
        self,
        name="rental_competition.train.npz",
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
        # Assuming train.data contains the features and train.target contains the target variable

        feature_names = [
            "season",
            "year",
            "month",
            "hour",
            "holiday",
            "day_of_week",
            "working_day",
            "weather",
            "temperature",
            "feeling_temperature",
            "relative_humidity",
            "windspeed",
        ]

        binary_features = ["holiday", "working_day"]
        categorical_features = [
            "season",
            "year",
            "month",
            "hour",
            "day_of_week",
            "weather",
        ]
        continuous_features = [
            "temperature",
            "feeling_temperature",
            "relative_humidity",
            "windspeed",
        ]

        # Feature engineering pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                # One-hot encode categorical features
                ("cat", OneHotEncoder(sparse_output=False), categorical_features),
                # No transformation for binary features (we keep them as is)
                ("bin", "passthrough", binary_features),
                # Standard scaling for continuous features
                ("num", StandardScaler(), continuous_features),
            ]
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ]
        )
        df_train = pd.DataFrame(train.data, columns=feature_names)
        df_train = pipeline.fit_transform(df_train)

        X_train, X_val, y_train, y_val = train_test_split(
            df_train, train.target, test_size=0.1, random_state=args.seed
        )

        # Create a linear regression model

        model = Lasso(0.16768329368110083, max_iter=5000)

        # Train the model
        model.fit(X_train, y_train)

        # Evaluate the model
        predictions = model.predict(X_val)
        rmse = root_mean_squared_error(y_val, predictions)
        print(f"Validation Mean Squared Error: {rmse}")

        # TODO: Train a model on the given dataset and store it in `model`.

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump((model, pipeline), model_file)

    else:
        # Use the model and return test set predictions, either as a Python list or a NumPy array.
        print("Predicting")
        feature_names = [
            "season",
            "year",
            "month",
            "hour",
            "holiday",
            "day_of_week",
            "working_day",
            "weather",
            "temperature",
            "feeling_temperature",
            "relative_humidity",
            "windspeed",
        ]
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model, pipeline = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions.
        df_test = pd.DataFrame(test.data, columns=feature_names)
        df_test = pipeline.transform(df_test)
        predictions = model.predict(df_test)

        return predictions


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)


# feature_names = [
#     "season",
#     "year",
#     "month",
#     "hour",
#     "holiday",
#     "day_of_week",
#     "working_day",
#     "weather",
#     "temperature",
#     "feeling_temperature",
#     "relative_humidity",
#     "windspeed",
# ]

# binary_features = ["holiday", "working_day"]
# categorical_features = ["season", "year", "month", "hour", "day_of_week", "weather"]
# continuous_features = [
#     "temperature",
#     "feeling_temperature",
#     "relative_humidity",
#     "windspeed",
# ]

# # Feature engineering pipeline
# preprocessor = ColumnTransformer(
#     transformers=[
#         # One-hot encode categorical features
#         ("cat", OneHotEncoder(sparse_output=False), categorical_features),
#         # No transformation for binary features (we keep them as is)
#         ("bin", "passthrough", binary_features),
#         # Standard scaling for continuous features
#         ("num", StandardScaler(), continuous_features),
#     ]
# )

# pipeline = Pipeline(
#     steps=[
#         ("preprocessor", preprocessor),
#         ("poly", PolynomialFeatures(degree=2, include_bias=False)),
#     ]
# )


# df_train = pd.DataFrame(train.data, columns=feature_names)

# df_train = pipeline.fit_transform(df_train)
# X_train, X_val, y_train, y_val = train_test_split(
#     df_train, train.target, test_size=0.1, random_state=42
# )

# rmses = []
# best_rmse = np.inf
# for lambda_ in np.geomspace(0.01, 10, num=500):
#     model = Lasso(alpha=lambda_, max_iter=5000)
#     model.fit(X_train, y_train)
#     predictions = model.predict(X_val)
#     rmse = np.sqrt(mean_squared_error(y_val, predictions))
#     rmses.append(rmse)
#     if rmse < best_rmse:
#         best_rmse = rmse
#         best_lambda = lambda_

# print(f"Best lambda: {best_lambda}, RMSE: {best_rmse}")


# # Train the model
# model.fit(X_train, y_train)

# # Evaluate the model
# predictions = model.predict(X_val)


# # Evaluate model using cross-validation (using RMSE as scoring)
# rmse_scores = np.sqrt(
#     -cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv=5)
# )
# print(f"Cross-validated RMSE: {np.mean(rmse_scores)}")
