"""
Functionss for drawing
JCA
Vaico
"""
import matplotlib.pyplot as plt
from os import path, makedirs
import numpy as np

def histogram(metric, metric_name, metric_avg, label, label_save_path):
    """Save simple histogram plot"""
    fig = plt.figure()
    _ = plt.hist(metric[metric_name], bins='auto')  # arguments are passed to np.histogram
    plt.axvline(metric[metric_avg], color='k', linestyle='dashed', linewidth=1)
    plt.title('{} distribution of {}'.format(metric_name.capitalize(), label))
    if label_save_path:
        fig.savefig(path.join(label_save_path, '{}_histogram.png'.format(metric_name)))


def confusion_matrix(results, label_save_path, label=None):
    """Create and save confusion matrix. If label is None create multi-class matrix"""
    if label:
        true_pos = 0
        false_neg = 0
        false_pos = 0
        true_neg = 0
        for res in results[label]:
            true_pos += res['true_pos'] if 'true_pos' in res else true_pos
            false_neg += res['false_neg'] if 'false_neg' in res else false_neg
            false_pos += res['false_pos'] if 'false_pos' in res else false_pos
            true_neg += res['true_neg'] if 'true_neg' in res else true_neg
    conf_mat = np.array([[true_pos, false_neg],[false_neg, true_neg]])
    fig = plt.figure()
    axes = fig.add_subplot(111)
    caxes = axes.matshow(conf_mat)
    fig.colorbar(caxes)
    plt.title('Confusion Matrix of {}'.format(label.capitalize()))

    axes.set_xticklabels(['']+['P','N'])
    axes.set_yticklabels(['']+['P','N'])
    if label_save_path:
        fig.savefig(path.join(label_save_path, '{}_cnf_mtrx.png'.format(label)))


def generate_plots(metrics, results, timestamp, save_path):
    """Generate and save plots from metrics"""
    print('Generating plots...')
    if save_path:
        saving_path = path.join(save_path, 'evaluation_plots', timestamp)
        makedirs(saving_path, exist_ok=True)
        print(' - Saving plots in: {}'.format(saving_path))

    for label, metric in metrics.items():
        print('   - Saving plots of {}'.format(label))
        if save_path:
            label_save_path = path.join(saving_path, label)
            makedirs(label_save_path, exist_ok=True)
        else:
            label_save_path = None
        histogram(metric, 'precision', 'avg_precision', label, label_save_path)
        histogram(metric, 'recall', 'avg_recall', label, label_save_path)
        histogram(metric, 'abs_error', 'avg_abs_error', label, label_save_path)
        confusion_matrix(results, label_save_path, label=label)
