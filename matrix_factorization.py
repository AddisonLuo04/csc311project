import numpy as np
from scipy.linalg import sqrtm

import matplotlib.pyplot as plt

from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    sparse_matrix_evaluate,
)


def svd_reconstruct(matrix, k):
    """Given the matrix, perform singular value decomposition
    to reconstruct the matrix.

    :param matrix: 2D sparse matrix
    :param k: int
    :return: 2D matrix
    """
    # First, you need to fill in the missing values (NaN) to perform SVD.
    # Fill in the missing values using the average on the current item.
    # Note that there are many options to do fill in the
    # missing values (e.g. fill with 0).
    new_matrix = matrix.copy()
    mask = np.isnan(new_matrix)
    masked_matrix = np.ma.masked_array(new_matrix, mask)
    item_means = np.mean(masked_matrix, axis=0)
    new_matrix = masked_matrix.filled(item_means)

    # Next, compute the average and subtract it.
    item_means = np.mean(new_matrix, axis=0)
    mu = np.tile(item_means, (new_matrix.shape[0], 1))
    new_matrix = new_matrix - mu

    # Perform SVD.
    Q, s, Ut = np.linalg.svd(new_matrix, full_matrices=False)
    s = np.diag(s)

    # Choose top k eigenvalues.
    s = s[0:k, 0:k]
    Q = Q[:, 0:k]
    Ut = Ut[0:k, :]
    s_root = sqrtm(s)

    # Reconstruct the matrix.
    reconst_matrix = np.dot(np.dot(Q, s_root), np.dot(s_root, Ut))
    reconst_matrix = reconst_matrix + mu
    return np.array(reconst_matrix)


def squared_error_loss(data, u, z):
    """Return the squared-error-loss given the data.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param u: 2D matrix
    :param z: 2D matrix
    :return: float
    """
    loss = 0
    for i, q in enumerate(data["question_id"]):
        loss += (data["is_correct"][i] - np.sum(u[data["user_id"][i]] * z[q])) ** 2.0
    return 0.5 * loss


def update_u_z(train_data, lr, u, z):
    """Return the updated U and Z after applying
    stochastic gradient descent for matrix completion.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param u: 2D matrix
    :param z: 2D matrix
    :return: (u, z)
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Randomly select a pair (user_id, question_id).
    i = np.random.choice(len(train_data["question_id"]), 1)[0]

    c = train_data["is_correct"][i]
    n = train_data["user_id"][i]
    q = train_data["question_id"][i]

    # compute the prediction for the selected pair.
    # predicted value = u_n^T z_m
    pred = np.dot(u[n], z[q])  

    # compute the difference between prediction and true values
    error = c - pred

    # update user vector (u_n) and question vector (z_m) using gradient descent
    # scaled by the learning rate (lr)
    u[n] += lr * error * z[q]  # ∂loss/∂u_n = -error * z[q]
    z[q] += lr * error * u[n]  # ∂loss/∂z_m = -error * u[n]

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return u, z


def als(train_data, val_data, k, lr, num_iteration):
    """Performs ALS algorithm, here we use the iterative solution - SGD
    rather than the direct solution.

    ** NEW ** for question 3e, also tracks losses of training data and
    validation data. Should not use validation data in training process,
    just to create the loss array

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :param lr: float
    :param num_iteration: int
    :return: 2D reconstructed Matrix and squared-error losses
    """
    # Initialize u and z
    u = np.random.uniform(
        low=0, high=1 / np.sqrt(k), size=(len(set(train_data["user_id"])), k)
    )
    z = np.random.uniform(
        low=0, high=1 / np.sqrt(k), size=(len(set(train_data["question_id"])), k)
    )

    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    
    train_losses = []
    val_losses = []

    # use update_u_z() to perform SGD for num_iteration many times
    for iteration in range(num_iteration):
        u, z = update_u_z(train_data, lr, u, z)
        
        # calculate and store losses periodically (every 10 000) iterations
        if iteration % 10000 == 0 or iteration == num_iteration - 1:
            train_loss = squared_error_loss(train_data, u, z)
            val_loss = squared_error_loss(val_data, u, z)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

    # reconstruct the matrix
    mat = np.dot(u, z.T)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return mat, train_losses, val_losses


def main():
    train_matrix = load_train_sparse("./data").toarray()
    train_data = load_train_csv("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    #####################################################################
    # TODO:                                                             #
    # (SVD) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    # list of k values to try and other variables
    # to keep track of the best k
    k_values = [2, 5, 10, 20, 50]
    best_svd_k = None
    best_svd_val_accuracy = 0

    print("Starting SVD reconstruction for different k values...")
    
    for k in k_values:
        # reconstruct the matrix using SVD
        reconstructed_matrix = svd_reconstruct(train_matrix, k)
        
        # evaluate performance on validation data
        val_accuracy = sparse_matrix_evaluate(val_data, reconstructed_matrix)
        print(f"k = {k}, Validation Accuracy = {val_accuracy:.4f}")
        
        # keep track of the best k
        if val_accuracy > best_svd_val_accuracy:
            best_svd_k = k
            best_svd_val_accuracy = val_accuracy

    print(f"\nBest k for SVD = {best_svd_k}, Best Validation Accuracy = {best_svd_val_accuracy:.4f}")
    
    # re-evaluate on test data with the best k
    best_reconstructed_matrix = svd_reconstruct(train_matrix, best_svd_k)
    test_accuracy = sparse_matrix_evaluate(test_data, best_reconstructed_matrix)
    print(f"Test Accuracy with best k = {best_svd_k}: {test_accuracy:.4f}")
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # (ALS) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    
    print("\nStarting ALS with SGD for different k values...")
    als_results = {}
    learning_rate = 0.06
    num_iterations = 70000

    # for storing losses for the best k* for plotting
    best_train_losses = []
    best_val_losses = []

    for k in k_values:
        reconstructed_matrix, train_losses, val_losses = als(
            train_data, val_data, k, learning_rate, num_iterations
        )
        val_accuracy = sparse_matrix_evaluate(val_data, reconstructed_matrix)
        als_results[k] = val_accuracy

        print(f"k = {k}, Final Validation Accuracy = {val_accuracy:.4f}")

        # save losses for the best k*
        if k == max(als_results, key=als_results.get):
            best_train_losses = train_losses
            best_val_losses = val_losses

    # select the best k based on validation accuracy
    best_k_als = max(als_results, key=als_results.get)
    print(f"\nBest k for ALS = {best_k_als}, Best Validation Accuracy = {als_results[best_k_als]:.4f}")

    # evaluate the test accuracy for the best k
    reconstructed_matrix, _, _ = als(train_data, val_data, best_k_als, learning_rate, num_iterations)
    test_accuracy_als = sparse_matrix_evaluate(test_data, reconstructed_matrix)
    print(f"Test Accuracy (ALS) with best k = {best_k_als}: {test_accuracy_als:.4f}")

    # plot losses for the best k
    iterations = list(range(0, num_iterations + 1, 10000))
    plt.plot(iterations, best_train_losses, label="Training Loss")
    plt.plot(iterations, best_val_losses, label="Validation Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Squared-Error Loss")
    plt.title(f"ALS Losses Over Iterations (k={best_k_als})")
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
