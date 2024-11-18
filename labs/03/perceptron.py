#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--data_size", default=100, type=int, help="Data size")
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
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# If you add more arguments, ReCodEx will keep them with your default values.


# generator = np.random.RandomState(42)
# data, target = sklearn.datasets.make_classification(
#     n_samples=100,
#     n_features=2,
#     n_informative=2,
#     n_redundant=0,
#     n_clusters_per_class=1,
#     flip_y=0,
#     class_sep=2,
#     random_state=42,
# )
# target = 2 * target - 1


# import matplotlib.pyplot as plt

# # Scatter plot with color mapping based on target values
# plt.scatter(data[:, 0], data[:, 1], c=target, cmap="bwr", edgecolor="k")
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
# plt.title("Binary Classification Data")
# plt.colorbar(
#     label="Class"
# )  # Optional: adds a color bar if target values are not just -1 and 1
# plt.show()

# data = np.hstack([data, np.ones((data.shape[0], 1))])
# data.shape
# weights = np.zeros(data.shape[1])


# done = False
# while not done:
#     done = True  # Assume we are done unless we find a misclassification
#     permutation = generator.permutation(data.shape[0])  # Random order of samples
#     print(permutation)

#     for i in permutation:
#         # Compute the prediction (sign of dot product between weights and data[i])
#         prediction = np.sign(np.dot(data[i], weights))

#         # If the prediction does not match the target
#         if prediction != target[i]:
#             # Update the weights: w = w + target[i] * data[i]
#             weights += target[i] * data[i]
#             done = False  # We found a misclassified sample, so we need another pass

#         # Plot the data and current decision boundary after each pass
#     plt.figure()
#     plt.scatter(data[:, 0], data[:, 1], c=target, cmap="bwr", alpha=0.7)

#     # Compute decision boundary based on weights
#     xs = np.linspace(data[:, 0].min(), data[:, 0].max(), 100)
#     ys = -(weights[0] / weights[1]) * xs - (weights[-1] / weights[1])

#     plt.plot(xs, ys, color="black")
#     plt.title("Perceptron Decision Boundary Update")
#     plt.xlabel("Feature 1")
#     plt.ylabel("Feature 2")
#     plt.show()


def main(args: argparse.Namespace) -> np.ndarray:
    # Create a random generator with a given seed.
    generator = np.random.RandomState(args.seed)
    # Generate binary classification data with labels {-1, 1}.
    data, target = sklearn.datasets.make_classification(
        n_samples=args.data_size,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        flip_y=0,
        class_sep=2,
        random_state=args.seed,
    )
    target = 2 * target - 1

    # TODO: Append a constant feature with value 1 to the end of all input data.
    # Then we do not need to explicitly represent bias - it becomes the last weight.
    data = np.hstack([data, np.ones((data.shape[0], 1))])
    # Generate initial perceptron weights.
    weights = np.zeros(data.shape[1])

    done = False
    while not done:
        done = True  # Assume we are done unless we find a misclassification
        permutation = generator.permutation(data.shape[0])

        # TODO: Implement the perceptron algorithm, notably one iteration
        # over the training data in the order of `permutation`. During the
        # training data iteration, perform the required updates to the `weights`
        # for incorrectly classified examples. If all training instances are
        # correctly classified, set `done=True`, otherwise set `done=False`.
        for i in permutation:
            # Compute the prediction (sign of dot product between weights and data[i])
            prediction = np.sign(np.dot(data[i], weights))

            # If the prediction does not match the target
            if prediction != target[i]:
                # Update the weights: w = w + target[i] * data[i]
                weights += target[i] * data[i]
                done = False  # We found a misclassified sample, so we need another pass

        if args.plot and not done:
            import matplotlib.pyplot as plt

            if args.plot is not True:
                plt.gcf().get_axes() or plt.figure(figsize=(6.4 * 3, 4.8 * 3))
                plt.subplot(3, 3, 1 + len(plt.gcf().get_axes()))
            plt.scatter(data[:, 0], data[:, 1], c=target)
            xs = np.linspace(*plt.gca().get_xbound() + (50,))
            ys = np.linspace(*plt.gca().get_ybound() + (50,))
            plt.contour(
                xs, ys, [[[x, y, 1] @ weights for x in xs] for y in ys], levels=[0]
            )
            (
                plt.show()
                if args.plot is True
                else plt.savefig(args.plot, transparent=True, bbox_inches="tight")
            )

    return weights


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    weights = main(main_args)
    print("Learned weights", *("{:.2f}".format(weight) for weight in weights))
