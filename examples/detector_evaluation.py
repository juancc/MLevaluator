from MLinference.architectures import Yolo4

from MLevaluator.evaluate import evaluate

dataset = '/misdoc/datasets/baluarte/00034/annotation.json'
save_path = '/home/juanc/tmp/model_evaluation/personas'

labels = {0: 'persona'}
model = Yolo4.load('/misdoc/vaico/architectures/yolov4_tflite/checkpoints/yolov4_custom_v2.tflite',
                   labels=labels, input_size=608)

evaluate(model, dataset,
         save_path=save_path,
         labels = labels,
         parents=[None],
         percentage=0.1,
         debug=False,
         iou_threshold=0.25)