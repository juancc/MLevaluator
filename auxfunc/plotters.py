"""
Functionss for drawing
JCA
Vaico
"""
import matplotlib.pyplot as plt
from os import path, makedirs
import numpy as np

def histogram(metric, metric_name, metric_avg, label, label_save_path, bins='auto', metric_nickname=None):
    """Save simple histogram plot"""
    fig = plt.figure()
    axes = fig.add_subplot(111)
    _ = plt.hist(metric[metric_name], bins=bins)  # arguments are passed to np.histogram
    plt.axvline(metric[metric_avg], color='k', linestyle='dashed', linewidth=1)

    metric_nickname = metric_nickname.capitalize() if metric_nickname else metric_name.capitalize()
    plt.title('{} distribution of {} per image'.format(metric_nickname, label))

    axes.set_xlabel(metric_nickname)
    axes.set_ylabel('No. images')

    if label_save_path:
        fig.savefig(path.join(label_save_path, '{}_histogram.png'.format(metric_name)))

def pie(metric, metric_name, labels, save_path, title=None):
    """Create and save simple pie chart"""
    sizes = []
    labels_names = []
    for l in labels.keys():
        total = sum(metric[l][metric_name])
        sizes.append(total)
        labels_names.append(l)
    fig, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels_names, autopct='%1.1f%%',
            shadow=False)
    ax1.axis('equal')
    if title: plt.title(title)

    if save_path:
        fig.savefig(path.join(save_path, '{}_pie.png'.format(metric_name)))

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
    conf_mat = np.array([[true_pos, false_neg],[false_pos, true_neg]])
    fig = plt.figure()
    axes = fig.add_subplot(111)
    caxes = axes.matshow(conf_mat)
    fig.colorbar(caxes)
    plt.title('Confusion Matrix of {}'.format(label.capitalize()))

    axes.set_xticklabels(['']+['P','N'])
    axes.set_yticklabels(['']+['P','N'])
    axes.set_ylabel('Actual')
    axes.set_xlabel('Predicted')
    if label_save_path:
        fig.savefig(path.join(label_save_path, '{}_cnf_mtrx.png'.format(label)))

def classes_confusion(classes_matrix, labels, save_path):
    """Plot confusion matrix of classes"""
    labels = [k for k, v in sorted(labels.items(), key=lambda item: item[1])]
    fig = plt.figure()
    axes = fig.add_subplot(111)
    caxes = axes.matshow(classes_matrix)
    fig.colorbar(caxes)
    plt.title('Classes Confusion Matrix')
    axes.set_xticklabels([''] + labels)
    axes.set_yticklabels([''] + labels)
    axes.set_ylabel('Actual')
    axes.set_xlabel('Predicted')
    if save_path:
        fig.savefig(path.join(save_path, 'classes_matrix.png'))

def generate_plots(metrics, results, classes_matrix, labels, save_path):
    """Generate and save plots from metrics"""
    print('Generating plots...')
    if save_path:
        saving_path = path.join(save_path, 'plots')
        makedirs(saving_path, exist_ok=True)
        print(' - Saving plots in: {}'.format(saving_path))

    for label, metric in metrics.items():
        if save_path:
            print('   - Saving plots of {}'.format(label))
            label_save_path = path.join(saving_path, label)
            makedirs(label_save_path, exist_ok=True)
        else:
            label_save_path = None
        if len(metric['precision'])>1: # There are distribution of metrics
            histogram(metric, 'precision', 'avg_precision', label, label_save_path, bins=20)
            histogram(metric, 'recall', 'avg_recall', label, label_save_path, bins=20)
            histogram(metric, 'f1', 'avg_f1', label, label_save_path, bins=20)
            histogram(metric, 'abs_error', 'avg_abs_error',
                      label, label_save_path, bins=20, metric_nickname='Absolute Error')
            histogram(metric, 'norm_abs_error', 'norm_avg_abs_error',
                      label, label_save_path, bins=20, metric_nickname='Normalized Absolute Error')
            histogram(metric, 'objs_per_image', 'avg_objs_per_image',
                      label, label_save_path, metric_nickname='Objects per Image', bins=20)

        confusion_matrix(results, label_save_path, label=label)
    if classes_matrix is not None:
        classes_confusion(classes_matrix, labels, save_path)

    pie(metrics, 'objs_per_image', labels, save_path, title='Dataset composition')