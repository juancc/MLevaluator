"""
Evaluators
Functions for performing the evaluation of model with the dataset depending of the type:
 - Classification
 - Detection

 JCA
 Vaico
"""
import json
from os import path, makedirs
from random import random

from tqdm import tqdm
import cv2 as cv
import numpy as np

from MLgeometry import creator, Object
from MLdrawer.drawer import draw

from MLevaluator.auxfunc.cropper import crop_rect

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

def update_count(count, label):
    if label in count:
        count[label] += 1
    else:
        count[label] = 1

def cascade_to_detection_labels(objects, model, labels):
    """Convert Cascade labels to detection labels.
    Add a new object with the geometry of detector and the label of the subobject.
    Detector object is removed for avoid confusing results"""
    new_objects = []
    for o in objects:
        o.label = o.label.lower()
        if o.label in model.sub_models.keys() and o.subobject:
            for sub_obj in o.subobject:
                sub_obj = sub_obj[0] if isinstance(sub_obj, list) else sub_obj # Classifier return prediction in a list
                sub_obj.label = sub_obj.label.lower()
                if sub_obj.label in labels.keys():
                    new_objects.append(
                        Object(
                            geometry =o.geometry,
                            label = sub_obj.label,
                            score = sub_obj.score,
                            subobject = None
                        )
                    )
    return new_objects

def posterior_to_detection_labels(objects, model, labels):
    """Convert Posterior labels to detection labels.
        Add a new object with the geometry of detector and the label of the subobject.
        Detector object is removed for avoid confusing results"""
    new_objects = []
    for o in objects:
        o.label = o.label.lower()
        if o.label in model.labels.keys() and o.subobject:
            for sub_obj in o.subobject:
                sub_obj = sub_obj[0] if isinstance(sub_obj, list) else sub_obj  # Classifier return prediction in a list
                sub_obj.label = sub_obj.label.lower()
                if sub_obj.label in model.labels[o.label]:
                    sub_obj.label = sub_obj.label.lower()
                    new_objects.append(
                        Object(
                            geometry=o.geometry,
                            label=sub_obj.label,
                            score=sub_obj.score,
                            subobject=None
                        )
                    )
    return new_objects



def detection(dataset, model, labels, iou_threshold, debug, save_path, mode='detection'):
    """Compute evaluation results of detection task. It is not computed the confusion matrix directly as miss placed
    geometries will not be possible to assign to any other label and also objects could not be predicted.
    True Positive: Object Intercept (IOU>threshold) and same label
    True Negatives: Don't apply for this metric. As any miss placed label that are not from the same class will count.
    False Positives: Extra predictions per class
    False Negatives: Missing predictions per class

    If mode=='cascade': labels of the sub-models will be treated as detector objects
    """
    results = {}
    n = len(labels)
    conf_matrix = np.zeros((n, n)) if n>1 else None  # With trues at Y (vertical axis)

    if save_path:
        saving_path = path.join(save_path, 'predictions')
        makedirs(saving_path, exist_ok=True)
        print(' - Saving prediction examples in: {}'.format(saving_path))

    i=0
    for observation in tqdm(dataset, total=len(dataset)):
        observation = json.loads(observation)
        im = cv.imread(observation['frame_id'])

        true_objs = creator.from_dict(observation['objects'])
        preds = model.predict(im)
        eval_preds = list(preds)
        if mode == 'cascade':
            true_objs = cascade_to_detection_labels(true_objs, model, labels)
            preds = cascade_to_detection_labels(preds, model, labels)
            eval_preds = cascade_to_detection_labels(eval_preds, model, labels)
        elif mode== 'posterior':
            true_objs = posterior_to_detection_labels(true_objs, model, labels)
            preds = posterior_to_detection_labels(preds, model, labels)
            eval_preds = posterior_to_detection_labels(eval_preds, model, labels)

        obs_results = {}
        predictions_per_class = {}  # Number of predictions per class on image
        predictions_counted = False  # Count the number of predictions only once
        true_per_class = {} # Number of true objects per class on image
        true_predicted_objects = [] # prediction objects that are true. Avoid double counting near elements
        for true in true_objs:
            true_predicted = False  # True already predicted
            true.label = true.label.lower()
            if true.label in labels.keys():  # Model is trained to predict that label
                update_count(true_per_class, true.label)# Count objects on image
                # Search if model predicted that object

                for p in eval_preds:
                    p.label = p.label.lower()
                    if not predictions_counted: # Count the number of predictions per class (only once)
                        update_count(predictions_per_class, p.label)
                    iou = p.geometry.iou(true.geometry)
                    if iou > iou_threshold:
                        if conf_matrix is not None:
                            pred_idx = labels[p.label]
                            true_idx = labels[true.label]
                            conf_matrix[true_idx, pred_idx] += 1
                        if true.label == p.label and not true_predicted and p not in true_predicted_objects:
                            add_result(obs_results, 'true_pos', p.label)
                            true_predicted = True
                            # Remove prediction for avoid double counting near elements
                            true_predicted_objects.append(p)


                predictions_counted = True # After make one loop for predictions

        # Count prediction even if not true objects on image
        if not predictions_counted:
            for p in preds:
                p.label = p.label.lower()
                update_count(predictions_per_class, p.label)

        # Add labels to observations results
        for label in predictions_per_class.keys():
            if label not in obs_results: obs_results[label] = {}
        for label in true_per_class.keys():
            if label not in obs_results: obs_results[label] = {}

        # Add missing false positives. Predictions away from trues
        for label, metrics in obs_results.items():
            metrics['true_pos'] = metrics['true_pos'] if 'true_pos' in metrics else 0
            metrics['true_neg'] = metrics['true_neg'] if 'true_neg' in metrics else 0
            if label not in predictions_per_class:  predictions_per_class[label] = 0
            if label not in true_per_class:  true_per_class[label] = 0

            metrics['false_pos'] = max(predictions_per_class[label] - metrics['true_pos'] , 0)
            metrics['false_neg'] = max(true_per_class[label] - metrics['true_pos'] , 0)
        update_global_results(results, obs_results)

        if random()>0.8:
            # Store prediction example
            sample_path = path.join(saving_path, str(i))
            draw(preds, im, draw_formats=FORMAT_PRED)
            draw(true_objs, im, draw_formats=FORMAT_TRUE)
            cv.imwrite(sample_path+'.jpg', im)
            with open(sample_path+'.json', 'w') as f:
                json.dump(obs_results, f)
            i += 1

        if debug:
            print(obs_results)
            draw(preds, im, draw_formats=FORMAT_PRED)
            draw(true_objs, im, draw_formats=FORMAT_TRUE)
            cv.imshow('Observations', im)
            k = cv.waitKey(0)
            if k == 113 or k == 27:  # q key, escape key
                break
            elif k == 32 or k == 83:  # space key, right arrow
                print('next')
                pass
    return results, conf_matrix

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


def classification(dataset, model, labels, debug, parents, save_path):
    """Evaluate model for classification problem. Dataset is given as a detection with subclasses.
    Model is triggered by a parent class. Evaluation is performed over the subclass of the detection
    """
    parents = [p.lower() for p in parents] if parents else None
    if not labels:
        print('For classification is required explicit labels (By the model or as function argument)')
        exit()

    if save_path:
        saving_path = path.join(save_path, 'predictions')
        makedirs(saving_path, exist_ok=True)
        print(' - Saving prediction examples in: {}'.format(saving_path))

    # Get conditions and weights from model to crop subregions
    print('Getting ROI conditions and weights. If not specified in the model defaults will be used.\n '
          'For specify custom, add model.roi_conditions and model. roi_weights')
    conditions = model.roi_conditions if hasattr(model, 'roi_conditions') else None
    weights = model.roi_weights if hasattr(model, 'roi_weights') else (0,0,1,1)
    print(' - Using ROI conditions: {}'.format(conditions))
    print(' - Using ROI weights: {}'.format(weights))

    conf_matrix = np.zeros((len(labels), len(labels)))  # With trues at Y (vertical axis)
    i=0
    for observation in tqdm(dataset, total=len(dataset)):
        observation = json.loads(observation)
        im = cv.imread(observation['frame_id'])
        true_objs = creator.from_dict(observation['objects'])
        for true in true_objs:
            true.label = true.label.lower()

            if true.label in parents and true.subobject:
                # Crop sub object area
                area = {
                    'xmin': true.geometry.xmin,
                    'ymin': true.geometry.ymin,
                    'xmax': true.geometry.xmax,
                    'ymax': true.geometry.ymax,
                }
                im_crop = crop_rect(im, area, weights, conditions)
                pred = model.predict(im_crop)[0]

                pred.label = pred.label.lower()
                for sub in true.subobject:
                    sub.label = sub.label.lower()
                    if sub.label in labels: # Dataset could contain labels for other problems
                        pred_idx = labels[pred.label]
                        true_idx = labels[sub.label]
                        conf_matrix[true_idx, pred_idx] += 1
                        if debug:
                            print('----')
                            print('Predictions:')
                            print(pred)
                            print('Classes Matrix:')
                            print(conf_matrix)
                            cv.imshow('Observations', im_crop)
                            k = cv.waitKey(0)
                            if k == 113 or k == 27:  # q key, escape key
                                break
                            elif k == 32 or k == 83:  # space key, right arrow
                                print('Next')
                                pass

                        if random() > 0.8 and save_path:
                            # Store prediction example
                            name = 'p:{}-t:{}.jpg'.format(pred.label, sub.label)
                            sample_path = path.join(saving_path, str(i))
                            cv.imwrite(sample_path + name, im_crop)
                            i += 1

    return process_classification(conf_matrix, labels), conf_matrix

def process_classification(results, labels):
    """Process confusion matrix of classificationr results.
    True values at columns
    :param results: (np.array) multiclass confusion martix
    Return a dict per class with true and false positives results
    """
    res = np.copy(results)
    processed = {}
    for label,i in labels.items():
        dg = np.array(np.diag(res))
        row = np.array(res[i,:])
        col = np.array(res[:,i])

        dg[i] = 0
        row[i] = 0
        col[i] = 0

        processed[label]= [{
            'true_pos': res[i,i],
            'true_neg': np.sum(dg),
            'false_neg': np.sum(row),
            'false_pos': np.sum(col)
        }]
    return processed

