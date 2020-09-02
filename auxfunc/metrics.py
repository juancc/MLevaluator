"""Calculate evaluation metrics
JCA
Vaico
"""
import numpy as np
def generate_metrics(results):
    """generate model metrics for label based on results"""
    print('Generating metrics...')
    metrics = {}
    for label, res in results.items():
        precision_label = precision(res)
        avg_precision = np.average(precision_label)
        std_precision = np.std(precision_label)  # standard deviation

        recall_label = recall(res)
        avg_recall = np.average(recall_label)
        std_recall = np.std(recall_label)  # standard deviation

        abs_error_label = abs_error(res)
        avg_abs_error = np.average(abs_error_label)
        std_abs_error = np.std(abs_error_label)  # standard deviation

        norm_abs_error_label = abs_error_norm(res)
        norm_avg_abs_error = np.average(norm_abs_error_label)
        norm_std_abs_error = np.std(norm_abs_error_label)  # standard deviation

        # Objects per image in test dataset
        objs_per_image = [m['true_pos'] + m['false_neg'] for m in res]
        avg_objs_per_image = np.average(objs_per_image)

        metrics[label] = {
            'precision': precision_label,
            'avg_precision': avg_precision,
            'std_precision': std_precision,

            'recall': recall_label,
            'avg_recall': avg_recall,
            'std_recall': std_recall,

            'abs_error': abs_error_label,
            'avg_abs_error': avg_abs_error,
            'std_abs_error': std_abs_error,

            'norm_abs_error': norm_abs_error_label,
            'norm_avg_abs_error': norm_avg_abs_error,
            'norm_std_abs_error': norm_std_abs_error,

            'objs_per_image': objs_per_image,
            'avg_objs_per_image': avg_objs_per_image,
        }
    print(' - Evaluation metrics of {}'.format(list(metrics)))
    return metrics


def precision(res):
    """Calculate the precision per observation of dict with confusion matrix elements.
    Precision: Percentage of correct predictions TP/(TP+FP)"""
    precision = []
    for m in res:
        try:
            precision.append(m['true_pos'] / (m['true_pos'] + m['false_pos']))
        except ZeroDivisionError:
            continue

    return precision

def recall(res):
    """Calculate the recall per observation of dict with confusion matrix elements.
    Recall or sensitivity: Percentage of predicted vs all ground true TP/(TP+FN)"""
    recall = []
    for m in res:
        try:
            recall.append(m['true_pos'] / (m['true_pos'] + m['false_neg']))
        except ZeroDivisionError:
            continue
    return recall

def abs_error(res):
    """Calculate the Absolute Error per image"""
    abs_error = []
    for m in res:
        try:
            abs_error.append( m['false_neg'] + m['false_pos'] )
        except ZeroDivisionError:
            continue
    return abs_error


def abs_error_norm(res):
    """Calculate the Normalized Absolute Error per image.
    Sum of errors (Miss-labeled) over the total of objects of the same class"""
    abs_error = []
    for m in res:
        try:
            abs_error.append(
                m['false_neg'] + m['false_pos']/ (m['true_pos']+ m['false_neg']) )
        except ZeroDivisionError:
            continue
    return abs_error


