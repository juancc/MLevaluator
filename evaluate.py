"""
Evaluate models that return MLgeometries with dataset in JSON-lines annotation file.
Recursively pass all the observations in dataset. Calculate metrics based predictions and true values

The model type is define if model run over all the image (main) or if run under some prediction (subobject)

PARAMS:
* model_config (JSON file)
{
    "path": path/to/.tflite
    "architecture": "Yolo4Lite",
    "labels": {idx:label}, # Optional, maybe passed in args
    "iou_threshold": 0.6, # Optional, depending of the model
    "parents": [parents,...] # Optional list with parents for subobject model
    "args": {
        "args_to_pass_to_model": "thresholds",
        "labels": {idx:label},
        ...
    }
}
If model in tflite is mandatory to specify the architecture

* dataset_path (JSON file)
Path to dataset json-lines annotation file

JCA
Vaico
"""
import argparse
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
from os import path, makedirs

from tqdm import tqdm
import cv2 as cv
import numpy as np

from MLgeometry import creator

from auxfunc.metrics import precision, recall, abs_error


FORMAT_PRED = {
    '*':{
        'style': 'classic',
        'fontScale': 0.3,
        'background_color': (0,0,255)
    }
}
FORMAT_TRUE = {
    '*':{
        'style': 'classic',
        'fontScale': 0.3,
        'background_color': (0,255,0)
    }
}

ARCH_TYPE = {
    'Yolo4Lite': 'detection'
}

def load_architecture(architecture):
    """Load architecture from String"""
    arch = None
    from sys import path
    path.append('/misdoc/vaico/architectures/Yolo4Lite')
    from Yolo4Lite import Yolo4Lite as arch
    return arch

def update_global_results(results, obs_result):
    """Add results from observations to global"""
    for label in list(obs_result):
        if label in results:
            results[label].append(obs_result[label])
        else:
            results[label] = [obs_result[label]]
    return results

def add_result(results, new_metric, label):
    """Add new data of the metric to results. Observation results
    :param new_metric: (str) such as true_pos, false_pos,
    """
    if label not in results:
        results[label] = {new_metric: 1}
    else:
        if new_metric in results[label]:
            results[label][new_metric] += 1
        else:
            results[label][new_metric] = 1
    return results

def evaluate(model_config_path, annotation_filepath, debug=False):
    """Iterate over the dataset observations. Make predictions and compare with true values
    Evaluate performance and save results"""
    if debug: # Visual debug of labels
        import sys
        sys.path.append('/misdoc/vaico/mldrawer/')
        from MLdrawer.drawer import draw

    print('Openining annotation file: {}'.format(annotation_filepath))
    with open(annotation_filepath, 'r') as f:
        dataset = f.readlines()
    print(' - Observation: {}'.format(len(dataset)))

    print('Loading model')
    with open(model_config_path, 'r') as f:
        model_config = json.load(f)

    arch_name = model_config['architecture']
    print(' - Loading Architecture: {}'.format(arch_name))
    arch = load_architecture(arch_name)
    model = arch.load(model_config['path'], **model_config['args'])

    model_type = 'main' if 'parent' not in model_config else 'subobject'
    prediction_type = ARCH_TYPE[arch_name] if arch_name in ARCH_TYPE else 'unknown'
    print(' - Model type: {}. Prediction type: {}'.format(model_type, prediction_type))

    print('Getting labels from model config')
    try:
        labels = model_config['labels'] if 'labels' in model_config else model_config['args']['labels']
    except KeyError:
        print('Model does not have labels. Some metrics will not be available')
        labels = None

    print(' - Model labels: {}'.format(labels if len(labels)<10 else len(labels)))

    print('Starting evaluation...')
    iou_threshold = model_config['iou_threshold'] if 'iou_threshold' in model_config else 0.35
    print(' - Using IOU threshold at: {}'.format(iou_threshold))
    # Results
    results = {} # per class confusion matrix elements

    start = time.time()
    i=0
    for observation in tqdm(dataset, total=len(dataset)):
        observation = json.loads(observation)
        im = cv.imread(observation['frame_id'])
        true_objs = creator.from_dict(observation['objects'])
        preds = model.predict(im)

        if model_type=='main' and prediction_type=='detection' and labels:
            obs_results = {}
            for true in true_objs:
                predictions_per_class = {}  # number of predictions per class
                predictions_counted = False
                true_predicted = False # True already predicted
                if true.label in labels.values(): # Model is trained to predict that label
                    # Search if model predicted that object
                    # IOU < iou_threshold
                    for p in preds:
                        # count predictions per class only once
                        if not predictions_counted:
                            if p.label in predictions_per_class:
                                predictions_per_class[p.label] += 1
                            else:
                                predictions_per_class[p.label] = 1
                        iou = p.geometry.iou(true.geometry)
                        if iou > iou_threshold:
                            if true.label == p.label and not true_predicted: # true positive
                                add_result(obs_results, 'true_pos', p.label)
                                true_predicted = True
                            else:# false positive
                                add_result(obs_results, 'false_pos', p.label)
                    predictions_counted = True
                    if not true_predicted:# Object wasnt predicted
                        add_result(obs_results, 'false_neg', true.label)
            # Add missing false positives. Predictions away from trues
            for label, metrics in obs_results.items():
                metrics['false_pos'] =  metrics['false_pos']  if 'false_pos' in metrics else 0
                metrics['true_pos'] =  metrics['true_pos']  if 'true_pos' in metrics else 0
                metrics['false_neg'] =  metrics['false_neg']  if 'false_neg' in metrics else 0
                if label not in predictions_per_class:  predictions_per_class[label] = 0

                new_false_neg = abs(metrics['true_pos'] + metrics['false_pos'] - predictions_per_class[label])
                metrics['false_pos'] += new_false_neg
            update_global_results(results, obs_results)
        if debug:
            print(obs_results)
            draw(preds, im,  draw_formats=FORMAT_PRED)
            draw(true_objs, im,  draw_formats=FORMAT_TRUE)
            cv.imshow('Observations', im)
            k = cv.waitKey(0)
            if k == 113 or k == 27:  # q key, escape key
                break
            elif k == 32 or k == 83:  # space key, right arrow
                print('next')
                pass
        i += 1
        if i==50: break

    time_elapsed = time.time() - start
    print('Evaluation done in: {}s'.format(time_elapsed))
    metrics = generate_metrics(results)
    # metrics['evaluation_info']={
    #     'dataset': annotation_filepath,
    #     'observations': len(dataset),
    #     'elapsed_time': time_elapsed
    # }
    timestamp = save_data(model_config_path, metrics, results)
    generate_plots(metrics, results, timestamp, model_config_path)

def generate_plots(metrics, results, timestamp, model_config_path):
    """Generate and save plots from metrics"""
    print('Generating plots...')
    root, _ = path.split(model_config_path)
    saving_path = path.join(root, 'evaluation_plots', timestamp)
    makedirs(saving_path, exist_ok=True)
    print(' - Saving plots in: {}'.format(saving_path))

    for label, metric in metrics.items():
        print('   - Saving plots of {}'.format(label))
        label_save_path = path.join(saving_path, label)
        makedirs(label_save_path, exist_ok=True)
        histogram_plot(metric, 'precision', 'avg_precision', label, label_save_path)
        histogram_plot(metric, 'recall', 'avg_recall', label, label_save_path)
        histogram_plot(metric, 'abs_error', 'avg_abs_error', label, label_save_path)
        confusion_matrix_plot(results, label_save_path, label=label)

        
def histogram_plot(metric, metric_name, metric_avg, label, label_save_path):
    """Save simple histogram plot"""
    fig = plt.figure()
    _ = plt.hist(metric[metric_name], bins='auto')  # arguments are passed to np.histogram
    plt.axvline(metric[metric_avg], color='k', linestyle='dashed', linewidth=1)
    plt.title('{} distribution of {}'.format(metric_name.capitalize(), label))
    fig.savefig(path.join(label_save_path, '{}_histogram.png'.format(metric_name)))

def confusion_matrix_plot(results, label_save_path, label=None):
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
    # plt.matshow(conf_mat)
    axes = fig.add_subplot(111)
    caxes = axes.matshow(conf_mat)
    fig.colorbar(caxes)
    # plt.colorbar()
    # plt.title('Confusion Matrix of {}'.format(label.capitalize()))
    fig.savefig(path.join(label_save_path, '{}_cnf_mtrx.png'.format(label)))
    # plt.show()




def save_data(model_config_path, metrics, results):
    """Save results and metrics in JSON formart"""
    now = datetime.now()
    timestamp = str(datetime.timestamp(now)).replace('.', '')
    saving_path, _ = path.split(model_config_path)
    print('Saving Metrics in: {}'.format(saving_path))
    data_path = path.join(saving_path, 'evaluation_data', timestamp)
    print(' - Making folder at: {}'.format(data_path))
    makedirs(data_path, exist_ok=True)
    metrics_file = path.join(data_path, 'metrics.json')
    results_file = path.join(data_path, 'results.json')
    print(' - Saving: {}'.format('metrics.json'))
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f)
    print(' - Saving: {}'.format('results.json'))
    with open(results_file, 'w') as f:
        json.dump(results, f)
    return timestamp


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
            'objs_per_image': objs_per_image,
            'avg_objs_per_image': avg_objs_per_image,
        }
    print(' - Evaluation metrics of {}'.format(list(metrics)))
    return metrics


if __name__ == '__main__':
    print('-- Model Evaluation --')
    parser = argparse.ArgumentParser()
    parser.add_argument("model_config_path", help="Path to model configuration")
    parser.add_argument("annotation_filepath", help="Path to dataset annotation json-lines file")
    parser.add_argument("-d", "--debug", help="Start visual debug of predictions and true labels", action='store_true')
    args = parser.parse_args()
    evaluate(args.model_config_path, args.annotation_filepath, args.debug)


