#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--classes", default=10, type=int, help="Number of classes to use")
parser.add_argument(
    "--epochs", default=10, type=int, help="Number of SGD training epochs"
)
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument(
    "--recodex", default=False, action="store_true", help="Running in ReCodEx"
)
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument(
    "--test_size",
    default=797,
    type=lambda x: int(x) if x.isdigit() else float(x),
    help="Test size",
)
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[np.ndarray, list[tuple[float, float]]]:
    # Create a random generator with a given seed.
    generator = np.random.RandomState(args.seed)

    # Load the digits dataset.
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)

    # Append a constant feature with value 1 to the end of all input data.
    # Then we do not need to explicitly represent bias - it becomes the last weight.
    data = np.pad(data, [(0, 0), (0, 1)], constant_values=1)

    # Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target = (
        sklearn.model_selection.train_test_split(
            data, target, test_size=args.test_size, random_state=args.seed
        )
    )

    # Generate initial model weights.
    weights = generator.uniform(
        size=[train_data.shape[1], args.classes], low=-0.1, high=0.1
    )

    for epoch in range(args.epochs):
        permutation = generator.permutation(train_data.shape[0])
        # TODO: Process the data in the order of `permutation`. For every
        # `args.batch_size` of them, average their gradient, and update the weights.
        # You can assume that `args.batch_size` exactly divides `train_data.shape[0]`.
        #
        # Note that you need to be careful when computing softmax because the exponentiation
        # in softmax can easily overflow. To avoid it, you should use the fact that
        # $softmax(z) = softmax(z + any_constant)$ and compute $softmax(z) = softmax(z - maximum_of_z)$.
        # That way we only exponentiate non-positive values, and overflow does not occur.

        # Process data in minibatches
        for i in range(0, train_data.shape[0], args.batch_size):
            batch_indices = permutation[i : i + args.batch_size]
            batch_data, batch_target = (
                train_data[batch_indices],
                train_target[batch_indices],
            )

            # Forward pass
            logits = batch_data @ weights
            logits -= np.max(logits, axis=1, keepdims=True)  # Stability fix
            exp_logits = np.exp(logits)
            softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

            # Cross-entropy loss and gradient
            one_hot_target = np.zeros_like(softmax_probs)
            one_hot_target[np.arange(args.batch_size), batch_target] = 1
            loss = (
                -np.sum(one_hot_target * np.log(softmax_probs + 1e-10))
                / args.batch_size
            )
            gradient = batch_data.T @ (softmax_probs - one_hot_target) / args.batch_size
            weights -= args.learning_rate * gradient

        # TODO: After the SGD epoch, measure the average loss and accuracy for both the
        # train test and the test set. The loss is the average MLE loss (i.e., the
        # negative log-likelihood, or cross-entropy loss, or KL loss) per example.
        train_logits = train_data @ weights
        train_logits -= np.max(train_logits, axis=1, keepdims=True)
        train_softmax = np.exp(train_logits) / np.sum(
            np.exp(train_logits), axis=1, keepdims=True
        )
        train_loss = -np.mean(
            np.log(train_softmax[np.arange(train_data.shape[0]), train_target] + 1e-10)
        )
        train_accuracy = np.mean(np.argmax(train_softmax, axis=1) == train_target)

        test_logits = test_data @ weights
        test_logits -= np.max(test_logits, axis=1, keepdims=True)
        test_softmax = np.exp(test_logits) / np.sum(
            np.exp(test_logits), axis=1, keepdims=True
        )
        test_loss = -np.mean(
            np.log(test_softmax[np.arange(test_data.shape[0]), test_target])
        )
        test_accuracy = np.mean(np.argmax(test_softmax, axis=1) == test_target)

        print(
            "After epoch {}: train loss {:.4f} acc {:.1f}%, test loss {:.4f} acc {:.1f}%".format(
                epoch + 1,
                train_loss,
                100 * train_accuracy,
                test_loss,
                100 * test_accuracy,
            )
        )

    return weights, [
        (train_loss, 100 * train_accuracy),
        (test_loss, 100 * test_accuracy),
    ]


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    weights, metrics = main(main_args)
    print(
        "Learned weights:",
        *(
            " ".join([" "] + ["{:.2f}".format(w) for w in row[:10]] + ["..."])
            for row in weights.T
        ),
        sep="\n"
    )
