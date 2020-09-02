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
from math import floor

from auxfunc.metrics import generate_metrics
from auxfunc.plotters import generate_plots
from auxfunc.evaluators import detection, classification


ARCH_TYPE = {
    'Yolo4': 'detection',
    'Yolo4Lite': 'detection',
    'KerasClassifiers': 'classification'
}


def evaluate(model, annotation_filepath, debug=False, labels=None, iou_threshold=0.35, save_path=None, parents=None, percentage=1.0):
    """Iterate over the dataset observations. Make predictions and compare with true values
    Evaluate performance and save results
    :param annotation_filepath: (str or list) list of annotation paths or annotation path
    :param parents: (list) parent prediction to trigger model
    :param percentage: (float) percentage used of the dataset for evaluation. 0 > percentage <= 1
    """
    print(' -- Model Evaluator --')
    print('Openining annotation file: {}'.format(annotation_filepath))
    dataset = []
    if isinstance(annotation_filepath, list):
        for ann in annotation_filepath:
            with open(ann, 'r') as f:
                d = f.readlines()
                dataset += d
    else:
        with open(annotation_filepath, 'r') as f:
            dataset = f.readlines()
    print(' - Observation: {}'.format(len(dataset)))
    # Percentage of dataset
    p = floor(percentage * len(dataset))
    dataset = dataset[:p]
    print(' - Using: {} of the dataset. Observations to evaluate: {}'.format(percentage, len(dataset)))

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
    else:
        print('Model labels replaced with given labels')
        model.labels = labels

    if isinstance(labels, list):
        labels = {label.lower(): idx for idx, label in enumerate(labels)} if labels else None
    else:
        labels = {label.lower():idx for idx, label in labels.items()} if labels else None

    print(' - Model labels: {}'.format(labels if len(labels)<10 else len(labels)))

    print('Starting evaluation...')
    print(' - Using IOU threshold at: {}'.format(iou_threshold))
    # Results
    results = {} # per class confusion matrix elements
    start = time.time()
    if prediction_type=='detection':
        results, classes_matrix = detection(dataset, model, labels, iou_threshold, debug)
    elif prediction_type=='classification':
        results, classes_matrix = classification(dataset, model, labels, debug, parents)
    time_elapsed = time.time() - start
    print('Evaluation done in: {}s'.format(time_elapsed))
    metrics = generate_metrics(results)

    if save_path:
        now = datetime.now()
        timestamp = str(datetime.timestamp(now)).replace('.', '')
        save_path = path.join(save_path, 'evaluation-'+timestamp)

        generate_plots(metrics, results, classes_matrix, labels, save_path)
        # Evaluation metadata
        metrics['evaluation_info'] = {
                'dataset': annotation_filepath,
                'observations': len(dataset),
                'elapsed_time': time_elapsed
            }
        save_data(metrics, results, save_path)
    else:
        print('Data will not be saved. save_path not specified')
    return metrics, results


def save_data(metrics, results, saving_path):
    """Save results and metrics in JSON formart"""
    print('Saving Metrics in: {}'.format(saving_path))
    data_path = path.join(saving_path, 'data')
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



if __name__ == '__main__':
    # from sys import path as sys_path
    # import sys
    # sys.path.append('/misdoc/vaico/MLinference')
    from MLinference.architectures import Yolo4

    dataset= '/misdoc/datasets/baluarte/00034/annotation.json'
    model_path = '/misdoc/vaico/architectures/yolov4_tflite/checkpoints/yolov4_custom_v2.tflite'
    labels = {0:'persona'}
    save_path = '/home/juanc/tmp/model_evaluation/personas00034'
    model = Yolo4.load(model_path, labels=labels, input_size=608)


    # sys_path.append('/misdoc/vaico/architectures/kerasclassifiers/')
    # from kerasClassifiers.KerasClassifiers import KerasClassifiers
    # dataset = '/misdoc/datasets/baluarte/00025/annotation.json'
    # model_path = '/home/juanc/Downloads/resnet_imageAI_arnes_v1.ml'
    # labels = ['con arnes','sin arnes']
    # save_path = '/home/juanc/tmp/model_evaluation/arnes'

    # model_path = '/home/juanc/Downloads/resnet50_ai_helmets_v1.ml'
    # labels = ['con casco','sin casco']
    # model_path = '/misdoc/vaico/models/Classifiers/PPE/helmet/helmets_resnet50-AI_fullbody-beta.ml'
    # save_path = '/home/juanc/tmp/model_evaluation/helmet_old'

    # model = KerasClassifiers.load(model_path)
    print(model.labels)

    evaluate(model, dataset,
             save_path=save_path,
             parents=['persona'],
             labels=labels,
             percentage=0.01,
             debug=False)