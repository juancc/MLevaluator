"""
Evaluate models that return MLgeometries with dataset in JSON-lines annotation file.
Recursively pass all the observations in dataset. Calculate metrics based predictions and true values

The model type is define if model run over all the image (main) or if run under some prediction (subobject)

For using from console:
PARAMS:
* model: ABCmodel or InferenceModel
* dataset_path (JSON file)
Path to dataset json-lines annotation file

JCA
Vaico
"""
import argparse
import json
import time
from datetime import datetime
from os import path, makedirs

from tqdm import tqdm
import cv2 as cv
import numpy as np

from MLgeometry import creator

from auxfunc.metrics import precision, recall, abs_error
from auxfunc.plotters import generate_plots

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

def evaluate(model, annotation_filepath, debug=False, labels=None, iou_threshold=0.35, save_path=None):
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

    arch_name = type(model).__name__
    model_type = 'main'
    prediction_type = ARCH_TYPE[arch_name] if arch_name in ARCH_TYPE else 'unknown'
    print(' - Model type: {}. Prediction type: {}'.format(model_type, prediction_type))

    print('Getting labels from model config')
    if not labels:
        try:
            labels = model.labels
        except AttributeError:
            print('Model does not have labels. Some metrics will not be available')
            labels = None

    print(' - Model labels: {}'.format(labels if len(labels)<10 else len(labels)))

    print('Starting evaluation...')
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
        if i==2: break

    time_elapsed = time.time() - start
    print('Evaluation done in: {}s'.format(time_elapsed))
    metrics = generate_metrics(results)

    if save_path:
        print('Saving data at: {}'.format(save_path))
        now = datetime.now()
        timestamp = str(datetime.timestamp(now)).replace('.', '')
        generate_plots(metrics, results, timestamp, save_path)
        # Evaluation metadata
        metrics['evaluation_info'] = {
                'dataset': annotation_filepath,
                'observations': len(dataset),
                'elapsed_time': time_elapsed
            }
        save_data(save_path, metrics, results, timestamp)
    else:
        print('Data will not be saved. save_path not specified')
    return metrics, results


def save_data(saving_path, metrics, results, timestamp):
    """Save results and metrics in JSON formart"""
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
    from sys import path as sys_path
    sys_path.append('/misdoc/vaico/architectures/Yolo4Lite')
    from Yolo4Lite import Yolo4Lite

    dataset= '/home/juanc/tmp/model_evaluation/annotation.json'
    model_path = '/home/juanc/tmp/model_evaluation/yolov4_custom_v2.tflite'
    save_path = '/home/juanc/tmp/model_evaluation/'
    model = Yolo4Lite.load(model_path, labels={0:'persona'}, input_size=608)

    evaluate(model, dataset, save_path=save_path)