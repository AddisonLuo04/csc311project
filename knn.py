import numpy as np
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
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


def main():
    sparse_matrix = load_train_sparse("./data").toarray()
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    k_values = [1, 6, 11, 16, 21, 26]
    best_k_user = None
    best_val_acc_user = -1
    best_k_item = None
    best_val_acc_item = -1
    val_accuracies_user = []
    val_accuracies_item = []

    # Evaluate user-based KNN
    for k in k_values:
        print(f"Evaluating user-based k = {k}")
        val_acc_user = knn_impute_by_user(sparse_matrix, val_data, k)
        val_accuracies_user.append(val_acc_user)

        if val_acc_user > best_val_acc_user:
            best_val_acc_user = val_acc_user
            best_k_user = k

    # Evaluate item-based KNN
    for k in k_values:
        print(f"Evaluating item-based k = {k}")
        val_acc_item = knn_impute_by_item(sparse_matrix, val_data, k)
        val_accuracies_item.append(val_acc_item)

        if val_acc_item > best_val_acc_item:
            best_val_acc_item = val_acc_item
            best_k_item = k

    print(f"Best k (user-based): {best_k_user} with validation accuracy: {best_val_acc_user}")
    print(f"Best k (item-based): {best_k_item} with validation accuracy: {best_val_acc_item}")

    # Evaluate test accuracy with the best k for each method
    test_acc_user = knn_impute_by_user(sparse_matrix, test_data, best_k_user)
    test_acc_item = knn_impute_by_item(sparse_matrix, test_data, best_k_item)

    print(f"Test accuracy (user-based) with k = {best_k_user}: {test_acc_user}")
    print(f"Test accuracy (item-based) with k = {best_k_item}: {test_acc_item}")

    # Plot the validation accuracy for both methods
    plt.plot(k_values, val_accuracies_user, label="User-based")
    plt.plot(k_values, val_accuracies_item, label="Item-based")
    plt.xlabel('k')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy vs k (User vs Item)')
    plt.legend()
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
