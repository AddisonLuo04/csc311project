import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
)

def bootstrap_sample(train_data):
    """
    Creates a bootstrapped sample from the training data.
    
    :param train_data: Training data dictionary.
    :return: Bootstrapped data dictionary.
    """
    n_samples = len(train_data["is_correct"])
    indices = np.random.choice(n_samples, n_samples, replace=True)
    return {key: np.array(train_data[key])[indices] for key in train_data}


def train_base_model(bootstrapped_data, max_depth, random_state):
    """
    Trains a single decision tree on the bootstrapped data.
    
    :param bootstrapped_data: Bootstrapped training data dictionary.
    :param max_depth: Maximum depth of the decision tree.
    :param random_state: Random state for reproducibility.
    :return: Trained decision tree model.
    """
    X_train = np.vstack([bootstrapped_data["user_id"], bootstrapped_data["question_id"]]).T
    y_train = bootstrapped_data["is_correct"]
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def predict_ensemble(models, data):
    """
    Predicts correctness using an ensemble of models.
    
    :param models: List of trained models.
    :param data: Data dictionary for prediction.
    :return: Ensemble predictions.
    """
    X = np.vstack([data["user_id"], data["question_id"]]).T
    preds = np.array([model.predict(X) for model in models])
    return np.round(preds.mean(axis=0))

def evaluate_model_accuracy(model, data):
    """
    Evaluates the accuracy of a trained model on a given dataset.
    
    :param model: The trained model (DecisionTreeClassifier).
    :param data: The dataset dictionary containing "user_id", "question_id", and "is_correct".
    :return: The accuracy of the model on the dataset.
    """
    # Prepare the features and target variable
    X = np.vstack([data["user_id"], data["question_id"]]).T
    y = data["is_correct"]
    
    # Predict the labels using the trained model
    predictions = model.predict(X)
    
    # Calculate and return the accuracy
    accuracy = accuracy_score(y, predictions)
    return accuracy

def ensemble(train_data, val_data, test_data, n_models, max_depth):
    """
    Implements an ensemble model using decision trees with bootstrapped training data.
    
    :param train_data: Training data dictionary.
    :param val_data: Validation data dictionary.
    :param test_data: Test data dictionary.
    :param n_models: Number of models in the ensemble.
    :param max_depth: Maximum depth of decision trees.
    :param max_features: Maximum number of features to 
    consider when splitting in decision trees.
    :return: Validation and test accuracies, and list of base model accuracies.
    """
    models = []
    model_accuracies = []

    # create and train base models
    for i in range(n_models):
        bootstrapped_data = bootstrap_sample(train_data)
        model = train_base_model(bootstrapped_data, max_depth=max_depth, random_state=i)
        models.append(model)

        accuracy_i = evaluate_model_accuracy(model, val_data)
        model_accuracies.append(accuracy_i)
        # print(f"Validation Accuracy for Decision Tree {i + 1}: {accuracy_i:.4f}")

    # evaluate the ensemble
    val_preds = predict_ensemble(models, val_data)
    test_preds = predict_ensemble(models, test_data)
    val_accuracy = accuracy_score(val_data["is_correct"], val_preds)
    test_accuracy = accuracy_score(test_data["is_correct"], test_preds)


    # # see how well the ensemble predicts the training data
    # # useful for seeing if the models are overfitting
    # train_preds = predict_ensemble(models, train_data)
    # train_accuracy = accuracy_score(train_data['is_correct'], train_preds)
    # print(f"Ensemble Training Accuracy for max d = {max_depth}: {train_accuracy:.4f}")

    return val_accuracy, test_accuracy, model_accuracies


def main():
    # load the data
    train_data = load_train_csv("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    # hyperparameter tuning for max_depth
    max_depth_values = [3, 5, 10, 15, 20]
    best_depth = None
    best_val_accuracy = 0
    n_models = 100

    print("Starting hyperparameter tuning for max_depth...")
    for max_depth in max_depth_values:
        print(f"Testing max_depth = {max_depth}")
        val_accuracy, _, _= ensemble(train_data, val_data, test_data, n_models=n_models, max_depth=max_depth)
        print(f"Validation Accuracy = {val_accuracy:.4f}")
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_depth = max_depth

    print(f"\nBest max_depth = {best_depth}, Best Validation Accuracy = {best_val_accuracy:.4f}")

    # final evaluation with best depth
    print("\nEvaluating ensemble with best max_depth...")
    final_val_accuracy, final_test_accuracy, final_model_accuracies = ensemble(train_data, val_data, test_data, n_models=n_models, max_depth=best_depth)
    print(f"Final Validation Accuracies for each individual model: {[round(acc, 4) for acc in final_model_accuracies]}")
    print(f"Final Validation Accuracy for Ensemble = {final_val_accuracy:.4f}")
    print(f"Final Test Accuracy = {final_test_accuracy:.4f}")


if __name__ == "__main__":
    main()