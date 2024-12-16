import numpy as np
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import csv
import os
import ast
import cProfile
import pstats
import time

from utils import (
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    sparse_matrix_evaluate,
)

def calculate_question_similarity(q1, q2, question_meta):
    """Calculate similarity between two questions based on shared subject IDs."""
    subjects1 = set(question_meta[q1])
    subjects2 = set(question_meta[q2])
    return len(subjects1 & subjects2) / len(subjects1 | subjects2) if subjects1 | subjects2 else 0

def calculate_student_similarity(s1, s2, student_meta):
    """Calculate similarity between two students based on metadata."""
    meta1, meta2 = student_meta.get(s1, {}), student_meta.get(s2, {})
    
    # Gender similarity
    gender_sim = 1 if meta1.get("gender") == meta2.get("gender") else 0
    
    # Age similarity (handle None gracefully)
    age1, age2 = meta1.get("age"), meta2.get("age")
    if age1 is not None and age2 is not None:
        age_sim = 1 / (1 + abs(age1 - age2))  # Smaller age difference yields higher similarity
    else:
        age_sim = 0  # Default to no similarity if either age is missing
    
    # Premium pupil similarity
    premium_sim = 1 if meta1.get("premium_pupil") == meta2.get("premium_pupil") else 0

    # Combine features with equal weights
    return (gender_sim + age_sim + premium_sim) / 3


def get_top_k_neighbors(row_index, col_index, matrix, k, is_user_based, question_meta, student_meta):
    """Retrieve the top k neighbors by directly finding feature-weighted similarity."""
    if is_user_based:
        similarities = [
            (i, calculate_student_similarity(row_index, i, student_meta))
            for i in range(matrix.shape[0]) if i != row_index
        ]
    else:
        similarities = [
            (j, calculate_question_similarity(col_index, j, question_meta))
            for j in range(matrix.shape[1]) if j != col_index
        ]
    # Sort by similarity (highest first) and take the top k
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[:k]
    return similarities

def precompute_similarities(matrix, is_user_based, question_meta, student_meta):
    """
    Precompute a similarity matrix for all user-user or item-item pairs.
    
    :param matrix: 2D sparse matrix (students x questions).
    :param is_user_based: Boolean, True if user-based similarity, False for item-based.
    :param question_meta: Dictionary mapping question_id to subject_id list.
    :param student_meta: Dictionary mapping user_id to metadata (e.g., gender, age).
    :return: 2D numpy array of precomputed similarities.
    """
    num_rows, num_cols = matrix.shape
    similarity_matrix = np.zeros((num_rows, num_rows) if is_user_based else (num_cols, num_cols))
    
    if is_user_based:
        # Map indices to user IDs for student metadata access
        index_to_user_id = {index: user_id for index, user_id in enumerate(sorted(student_meta.keys()))}
    else:
        # Map indices to question IDs for question metadata access
        index_to_question_id = {index: question_id for index, question_id in enumerate(sorted(question_meta.keys()))}

    for i in range(similarity_matrix.shape[0]):
        for j in range(i, similarity_matrix.shape[0]):  # Symmetric matrix, avoid duplicate calculations
            if is_user_based:
                # Map indices to user IDs
                user_i = index_to_user_id.get(i)
                user_j = index_to_user_id.get(j)
                if user_i is not None and user_j is not None:
                    # Pass user IDs to calculate similarity
                    sim = calculate_student_similarity(user_i, user_j, student_meta)
                else:
                    sim = 0  # Default similarity for missing metadata
            else:
                # Map indices to question IDs
                question_i = index_to_question_id.get(i)
                question_j = index_to_question_id.get(j)
                if question_i is not None and question_j is not None:
                    # Pass question IDs to calculate similarity
                    sim = calculate_question_similarity(question_i, question_j, question_meta)
                else:
                    sim = 0  # Default similarity for missing metadata
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim  # Ensure symmetry
    
    return similarity_matrix



def get_top_k_neighbors_from_cache(index, k, similarity_matrix):
    """
    Retrieve the top k neighbors from the precomputed similarity matrix.
    
    :param index: Row/column index for the target user or item.
    :param k: Number of neighbors to retrieve.
    :param similarity_matrix: Precomputed similarity matrix.
    :return: List of tuples (neighbor_index, similarity_score).
    """
    similarities = similarity_matrix[index]
    neighbors = [(i, sim) for i, sim in enumerate(similarities) if i != index]
    neighbors = sorted(neighbors, key=lambda x: x[1], reverse=True)[:k]
    return neighbors


def weighted_prediction(neighbors, row, col, matrix, is_user_based, global_mean=None):
    """
    Calculate the weighted prediction for a missing value.

    :param neighbors: List of (index, similarity) tuples for the top-k neighbors.
    :param row: The row index of the query point in the matrix.
    :param col: The column index of the query point in the matrix.
    :param matrix: The data matrix.
    :param is_user_based: Boolean, whether using user-based similarity.
    :return: Weighted prediction or fallback value (e.g., mean or default).
    """
    values = []
    weights = []
    
    for neighbor_index, similarity in neighbors:
        if is_user_based:
            value = matrix[neighbor_index, col]
        else:
            value = matrix[row, neighbor_index]

        if not np.isnan(value):  # Include only valid (non-NaN) values
            values.append(value)
            weights.append(similarity)

    values = np.array(values)
    weights = np.array(weights)
    
    # Handle the case where no valid neighbors were found
    if len(values) == 0:
        # Fallback to a default value, such as the global mean of the matrix
        if global_mean is None:
            global_mean = np.nanmean(matrix)  # Only calculate once
        return global_mean

    values = np.array(values)
    weights = np.array(weights)

    # Handle the case where all weights are zero or there are no valid neighbors
    if weights.sum() == 0:
        if global_mean is None:
            global_mean = np.nanmean(matrix)  # Only calculate once
        return global_mean

    # Compute the weighted average
    return np.dot(weights, values) / weights.sum()


def feature_weighted_knn(matrix, valid_data, k, question_meta, student_meta, is_user_based=True):
    """
    Perform feature-weighted KNN imputation and return the validation accuracy.
    
    :param matrix: 2D sparse matrix (students x questions), NaN for missing values.
    :param valid_data: Dictionary {user_id: list, question_id: list, is_correct: list}.
    :param k: Number of neighbors to consider.
    :param question_meta: Dictionary mapping question_id to subject_id list.
    :param student_meta: Dictionary mapping user_id to metadata (e.g., gender, age).
    :param is_user_based: Boolean flag to decide whether to use user-based or item-based similarity.
    :return: Validation accuracy as a float.
    """

    # Precompute similarity matrix
    similarity_matrix = precompute_similarities(matrix, is_user_based, question_meta, student_meta)
    
    # compute the global mean to use for NaN values
    global_mean = np.nanmean(matrix)

    # Main imputation loop
    imputed_matrix = matrix.copy()
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            if np.isnan(matrix[row, col]):  # Missing value
                if is_user_based:
                    neighbors = get_top_k_neighbors_from_cache(row, k, similarity_matrix)
                else:
                    neighbors = get_top_k_neighbors_from_cache(col, k, similarity_matrix)
                imputed_matrix[row, col] = weighted_prediction(neighbors, row, col, matrix, is_user_based, global_mean)
        # print(f"Processed row: {row}")
    
    # Evaluate the imputed matrix on validation data
    accuracy = sparse_matrix_evaluate(valid_data, imputed_matrix)
    return accuracy


def _load_meta_csv(path, key_col, value_col, parse_value=None):
    """
    A helper function to load metadata from a CSV file.

    :param path: str, Path to the CSV file.
    :param key_col: str, The name of the column to use as keys.
    :param value_col: str, The name of the column to use as values.
    :param parse_value: Callable, Optional function to parse the value column.
    :return: A dictionary mapping keys to values.
    """
    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))

    data = {}
    with open(path, "r") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            try:
                key = int(row[key_col])
                value = row[value_col]
                if parse_value:
                    value = parse_value(value)
                data[key] = value
            except (ValueError, KeyError):
                # Skip rows with invalid data.
                continue
    return data

def load_question_meta(root_dir="./data"):
    """
    Load question metadata as a dictionary.

    :param root_dir: str, Root directory of the metadata file.
    :return: A dictionary mapping question_id to a list of subject_ids.
    """
    path = os.path.join(root_dir, "question_meta.csv")
    return _load_meta_csv(
        path,
        key_col="question_id",
        value_col="subject_id",
        parse_value=lambda x: ast.literal_eval(x),
    )

def load_student_meta(root_dir="./data"):
    student_meta = {}

    # Define the path to your CSV file
    file_path = f"{root_dir}/student_meta.csv"
    current_year = 2024  # Hardcoded for age calculation
    
    try:
        with open(file_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            
            for row in reader:
                user_id = int(row["user_id"])
                gender = int(row["gender"]) if row["gender"] else None
                
                # Calculate age based on the year difference
                dob = row["data_of_birth"]
                if dob:
                    dob_year = int(dob[:4])  # Extract the year from the date string
                    age = current_year - dob_year
                else:
                    age = None  # If dob is missing, age is unknown
                
                # Parse premium_pupil
                premium_pupil = float(row["premium_pupil"]) if row["premium_pupil"] else None
                
                # Add the user metadata to the dictionary
                student_meta[user_id] = {
                    "gender": gender,
                    "age": age,  # Store the calculated age
                    "premium_pupil": premium_pupil
                }
                
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")

    return student_meta

def filter_valid_data(matrix, valid_data):
    max_user_id, max_question_id = matrix.shape
    filtered_data = {
        "user_id": [],
        "question_id": [],
        "is_correct": [],
    }
    for i in range(len(valid_data["user_id"])):
        if valid_data["user_id"][i] < max_user_id and valid_data["question_id"][i] < max_question_id:
            filtered_data["user_id"].append(valid_data["user_id"][i])
            filtered_data["question_id"].append(valid_data["question_id"][i])
            filtered_data["is_correct"].append(valid_data["is_correct"][i])
    return filtered_data

def add_trend_line(ax, x, y, color, label):
    z = np.polyfit(x, y, 2)  # 2nd-degree polynomial for a smooth trend line
    p = np.poly1d(z)
    x_fine = np.linspace(min(x), max(x), 500)
    y_fine = p(x_fine)
    ax.plot(x_fine, y_fine, color=color, linestyle="--", label=label)

def main():
    # Load data
    sparse_matrix = load_train_sparse("./data").toarray()
    valid_data = load_valid_csv("./data")
    # Load meta data
    question_meta = load_question_meta(root_dir="./data")
    student_meta = load_student_meta(root_dir="./data")

    # Use small matrix for demonstration:
    small_matrix = sparse_matrix[:500, :500]
    small_val_data = filter_valid_data(small_matrix, valid_data)

    # Set hyperparameters to try
    k_values = [1, 6, 11, 16, 21]
    is_user_based_values = [True, False] # True for user-based, False for item-based

    # Lists to store results for plotting
    user_based_accuracies = []
    item_based_accuracies = []
    user_based_times = []
    item_based_times = []

    for is_user_based in [True]:
        for k in [6]:
            print(f"Running feature-weighted KNN (k = {k}) with {'user' if is_user_based else 'item'}-based similarity...")
            
            # Start the timer
            start_time = time.time()

            accuracy = feature_weighted_knn(
                matrix=small_matrix,
                valid_data=small_val_data,
                k=k,
                question_meta=question_meta,
                student_meta=student_meta,
                is_user_based=is_user_based
            )

            # End the timer
            end_time = time.time()
            
            elapsed_time = end_time - start_time
            
            # Store the results
            if is_user_based:
                user_based_accuracies.append(accuracy)
                user_based_times.append(elapsed_time)
            else:
                item_based_accuracies.append(accuracy)
                item_based_times.append(elapsed_time)

            print(f"Validation Accuracy: {accuracy:.4f}")
            print(f"Time taken: {elapsed_time:.4f} seconds")


    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot for user-based
    im1 = axs[0].scatter(k_values, user_based_accuracies, c=user_based_times, cmap="viridis", s=100)
    add_trend_line(axs[0], k_values, user_based_accuracies, color="blue", label="Trend Line")
    axs[0].set_title("User-Based Similarity")
    axs[0].set_xticks(k_values)  # Set custom ticks
    axs[0].set_xlabel("k (Number of Neighbors)")
    axs[0].set_ylabel("Validation Accuracy")
    plt.colorbar(im1, ax=axs[0], label="Time Taken (seconds)")
    axs[0].legend()

    # Plot for item-based
    im2 = axs[1].scatter(k_values, item_based_accuracies, c=item_based_times, cmap="viridis", s=100)
    add_trend_line(axs[1], k_values, item_based_accuracies, color="red", label="Trend Line")
    axs[1].set_title("Item-Based Similarity")
    axs[1].set_xticks(k_values)  # Set custom ticks
    axs[1].set_xlabel("k (Number of Neighbors)")
    axs[1].set_ylabel("Validation Accuracy")
    plt.colorbar(im2, ax=axs[1], label="Time Taken (seconds)")
    axs[1].legend()

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig("knn_validation_vs_time.png")
    plt.show()



if __name__ == "__main__":
    # main()
    # Run your code with profiling
    cProfile.run('main()', 'profile_results.prof')

    # Load the profiling results
    p = pstats.Stats('profile_results.prof')

    # Sort by total time and print
    p.sort_stats('tottime').print_stats()
