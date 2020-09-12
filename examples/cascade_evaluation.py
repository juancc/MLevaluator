import sys
# sys.path.append('/misdoc/vaico/MLinference')
sys.path.append('/misdoc/vaico/architectures/kerasclassifiers/')
from kerasClassifiers.KerasClassifiers import KerasClassifiers
from MLinference.architectures import Yolo4
from MLinference.strategies import Cascade

from MLevaluator.evaluate import evaluate

dataset = '/misdoc/datasets/baluarte/00025/annotation.json'
save_path = '/home/juanc/tmp/model_evaluation/cascade'


labels_main_model = {0: 'persona'}
model_main = Yolo4.load('/misdoc/vaico/architectures/yolov4_tflite/checkpoints/yolov4_custom_v2.tflite',
                        labels=labels_main_model, input_size=608)

labels_submodel = ['con arnes', 'sin arnes']
model_classifier = KerasClassifiers.load('/home/juanc/Downloads/resnet_imageAI_arnes_v1.ml')

# Load models with evaluation labels and submodels with a unique ID
# The ID is used to refer the evaluation results
model = Cascade(
    main_model={
        'model': model_main,
        'labels': labels_main_model},
    sub_models={
        'persona': [
            {'model': model_classifier,
             'labels': labels_submodel,
             }]
    })


evaluate(model, dataset,
         save_path=save_path,
         parents=[None],
         percentage=0.2,
         debug=False)