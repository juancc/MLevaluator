"""
Evaluators
Functions for performing the evaluation of model with the dataset depending of the type:
 - Classification
 - Detection

 JCA
 Vaico
"""
import json

from tqdm import tqdm
import cv2 as cv

from MLgeometry import creator


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

def detection(dataset, model, labels, iou_threshold, debug):
    """Compute evaluation results of detection task"""
    if debug: # Visual debug of labels
        import sys
        sys.path.append('/misdoc/vaico/mldrawer/')
        from MLdrawer.drawer import draw

    results = {}
    i = 0
    for observation in tqdm(dataset, total=len(dataset)):
        observation = json.loads(observation)
        im = cv.imread(observation['frame_id'])
        true_objs = creator.from_dict(observation['objects'])
        preds = model.predict(im)

        # if model_type == 'main' and prediction_type == 'detection' and labels:
        obs_results = {}
        for true in true_objs:
            predictions_per_class = {}  # number of predictions per class
            predictions_counted = False
            true_predicted = False  # True already predicted
            if true.label in labels.values():  # Model is trained to predict that label
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
                        if true.label == p.label and not true_predicted:  # true positive
                            add_result(obs_results, 'true_pos', p.label)
                            true_predicted = True
                        else:  # false positive
                            add_result(obs_results, 'false_pos', p.label)
                predictions_counted = True
                if not true_predicted:  # Object wasnt predicted
                    add_result(obs_results, 'false_neg', true.label)
        # Add missing false positives. Predictions away from trues
        for label, metrics in obs_results.items():
            metrics['false_pos'] = metrics['false_pos'] if 'false_pos' in metrics else 0
            metrics['true_pos'] = metrics['true_pos'] if 'true_pos' in metrics else 0
            metrics['false_neg'] = metrics['false_neg'] if 'false_neg' in metrics else 0
            if label not in predictions_per_class:  predictions_per_class[label] = 0

            new_false_neg = abs(metrics['true_pos'] + metrics['false_pos'] - predictions_per_class[label])
            metrics['false_pos'] += new_false_neg
        update_global_results(results, obs_results)
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
        i += 1
        if i == 2: break
    return results

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


def classification(dataset, model, labels, debug, parents):
    """Evaluate model for classification problem. Dataset is given as a detection with subclasses.
    Model is triggered by a parent class. Evaluation is performed over the subclass of the detection"""
    i=0
    for observation in tqdm(dataset, total=len(dataset)):
        observation = json.loads(observation)
        im = cv.imread(observation['frame_id'])
        true_objs = creator.from_dict(observation['objects'])
        preds = model.predict(im)
        print(preds)
        exit()
        # if model_type == 'main' and prediction_type == 'detection' and labels:
        obs_results = {}
        # for true in true_objs: