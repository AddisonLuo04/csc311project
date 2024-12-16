import numpy as np
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from annoy import AnnoyIndex

from utils import (
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    sparse_matrix_evaluate,
)


def knn_impute_by_user(matrix, valid_data, k):
    """Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Transpose the matrix to treat questions as rows and students as columns
    item_matrix = matrix.T

    # Apply KNN imputer to the transposed matrix (item-based)
    nbrs = KNNImputer(n_neighbors=k)
    mat = nbrs.fit_transform(item_matrix)

    # Re-transpose the matrix back to its original form
    mat = mat.T

    # Evaluate accuracy on the validation data
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy (item-based): {}".format(acc))
    
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def compute_weights(distances, scheme="inverse", sigma=1.0, epsilon=1e-5):
    """Compute weights based on the given weighting scheme.
    
    :param distances: Array of distances.
    :param scheme: Weighting scheme ("inverse", "inverse_squared", "gaussian").
    :param sigma: Standard deviation for Gaussian kernel.
    :param epsilon: Small value to avoid division by zero.
    :return: Array of weights.
    """
    if scheme == "inverse":
        return 1 / (distances + epsilon)
    elif scheme == "inverse_squared":
        return 1 / (distances**2 + epsilon)
    elif scheme == "gaussian":
        return np.exp(-distances**2 / (2 * sigma**2))
    else:
        raise ValueError("Unknown weighting scheme: {}".format(scheme))


def knn_weighted_impute_with_annoy(matrix, valid_data, k, scheme="inverse"):
    n_students, n_questions = matrix.shape
    mat_filled = np.copy(matrix)

    # Fill missing values with KNN imputer before starting distance calculation
    nbrs = KNNImputer(n_neighbors=k)
    mat_filled = nbrs.fit_transform(mat_filled)

    # Use Annoy for fast approximate nearest neighbor search
    annoy_index = AnnoyIndex(n_questions, 'angular')  # angular distance is often faster
    for i in range(n_students):
        annoy_index.add_item(i, mat_filled[i, :])

    annoy_index.build(1)  # 10 trees is a good balance of speed and accuracy

    for i in range(n_students):
        for j in range(n_questions):
            if np.isnan(matrix[i, j]):
                # Find nearest neighbors
                neighbors = annoy_index.get_nns_by_item(i, k)

                # Get the corresponding values of the neighbors
                neighbor_values = mat_filled[neighbors, j]

                # Calculate the distances between the current student and the neighbors
                distances = np.array([euclidean_distances([mat_filled[i, :]], [mat_filled[neighbor, :]])[0][0] for neighbor in neighbors])

                # Compute weights using the given weighting scheme
                weights = compute_weights(distances, scheme=scheme)

                # Compute the weighted average of the neighbors' values for imputation
                weighted_sum = np.dot(weights, neighbor_values)
                weight_sum = np.sum(weights)

                # To avoid division by zero, ensure the weight sum is not zero
                if weight_sum > 0:
                    imputed_value = weighted_sum / weight_sum
                else:
                    imputed_value = 0  # or choose another default value

                # Assign the imputed value to the matrix
                mat_filled[i, j] = imputed_value

    return mat_filled


def main():
    sparse_matrix = load_train_sparse("./data").toarray()
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    k_values = [1, 6, 11, 16, 21, 26]
    best_k_weighted = None
    best_val_acc_weighted = -1
    val_accuracies_weighted = []

    # Evaluate weighted KNN with "inverse" weighting scheme
    for k in k_values:
        print(f"Evaluating weighted KNN (inverse) with k = {k}")
        # Change this line to use the correct function
        val_acc_weighted = knn_weighted_impute_with_annoy(
            sparse_matrix, val_data, k, scheme="inverse"
        )
        val_accuracies_weighted.append(val_acc_weighted)

        if val_acc_weighted > best_val_acc_weighted:
            best_val_acc_weighted = val_acc_weighted
            best_k_weighted = k

    # Print the best k and validation accuracy for weighted KNN
    print(f"Best k (weighted): {best_k_weighted} with validation accuracy: {best_val_acc_weighted}")

    # Evaluate test accuracy with the best k for weighted KNN
    test_acc_weighted = knn_weighted_impute_with_annoy(
        sparse_matrix, test_data, best_k_weighted, scheme="inverse"
    )
    print(f"Test accuracy (weighted) with k = {best_k_weighted}: {test_acc_weighted}")

    # Plot the validation accuracy for weighted KNN
    plt.plot(k_values, val_accuracies_weighted, label="Weighted (Inverse)")
    plt.xlabel('k')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy vs k (Weighted KNN)')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()