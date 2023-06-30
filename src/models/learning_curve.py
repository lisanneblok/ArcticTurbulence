from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np


def learning_curve(model, X_train, y_train):
    """
    Generates a learning curve to evaluate the performance of a model on the
    training data.

    Args:
        model: The trained model.
        X_train (array-like or sparse matrix): The input features of the
            training data.
        y_train (array-like): The target values of the training data.

    """
    # Create the learning curve
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, scoring='r2',
        train_sizes=np.linspace(0.1, 1.0, 10))

    # Calculate the mean and standard deviation of the train and test scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Plot the learning curve
    plt.figure(figsize=(10, 6))
    plt.title("Learning Curve")
    plt.xlabel("Training Examples")
    plt.ylabel("Score (R2)")
    plt.grid()

    plt.fill_between(train_sizes, train_mean - train_std,
                     train_mean + train_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std,
                     alpha=0.1, color="g")
    plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training Score")
    plt.plot(train_sizes, test_mean, 'o-', color="g",
             label="Cross-validation Score")

    plt.legend(loc="best")
    plt.show()
