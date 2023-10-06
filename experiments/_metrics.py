def classification_metrics(actual_result, predicted_result):
    """Calculate the precision, recall, and F1-score given sets of actual and predicted results.

    Parameters:
    - actual_result (set): The set of actual results (ground truth).
    - predicted_result (set): The set of predicted results.

    Returns:
    - accuracy (float): The accuracy of the predictions.
    - precision (float): The precision of the predictions.
    - recall (float): The recall of the predictions.
    - f1 (float): The F1-score of the predictions.
    """
    TP = len(actual_result.intersection(predicted_result))

    # False Positive (FP)
    FP = len(predicted_result.difference(actual_result))

    # False Negative (FN)
    FN = len(actual_result.difference(predicted_result))

    # Calculating Precision and Recall
    accuracy = TP / len(actual_result) if len(actual_result) != 0 else 0  # Corrected accuracy formula
    precision = TP / (TP + FP) if TP + FP != 0 else 0
    recall = TP / (TP + FN) if TP + FN != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0

    return accuracy, precision, recall, f1
