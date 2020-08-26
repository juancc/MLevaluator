"""Calculate evaluation metrics
JCA
Vaico
"""

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
    """Calculate the Absolute Error per observation of dict with confusion matrix elements.
    Absolute error: |pred-true|"""
    abs_error = []
    for m in res:
        try:
            abs_error.append(abs(m['true_pos'] - m['true_pos'] + m['false_neg']))
        except ZeroDivisionError:
            continue
    return abs_error