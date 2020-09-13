from MLinference.architectures import Yolo4
from MLinference.strategies import Multi, Posterior
from MLinference.architectures import UNet
from MLinference.architectures import OnEdge

from MLevaluator.evaluate import evaluate

# Evaluation Data
dataset = '/misdoc/datasets/baluarte/00034/annotation.json'
save_path = '/home/juanc/tmp/model_evaluation/posterior'
eval_percentage= 0.1
iou_threshold= 0.3
# using only for evaluation
labels = {
    'persona': ['lejos de borde', 'cerca de borde']
}

# Models parameters
detector_model_path = '/misdoc/vaico/architectures/yolov4_tflite/checkpoints/yolov4_custom_v2.tflite'
detector_labels = {0:'persona'}
input_size=608

edges_mask_path = '/home/juanc/Downloads/bordes_Unet-20200901.tflite'
edges_mask_labels = {0:'borde'}

edge_intereset_objects = ['persona']
edge_labels = {0:'lejos de borde', 1:'cerca de borde'}


# Instantiate models and evaluate
worker_detector = Yolo4.load(detector_model_path, labels=detector_labels, input_size=input_size)

edges_mask = UNet.load(edges_mask_path, labels=edges_mask_labels)
on_edge = OnEdge(None, interest_labels=edge_intereset_objects, mask_label=edges_mask_labels[0], labels=edge_labels)
main_models = Multi(models=[worker_detector, edges_mask])

model = Posterior(models=[main_models, on_edge])


evaluate(model, dataset,
         save_path=save_path,
         labels = labels,
         parents=[None],
         percentage=eval_percentage,
         debug=False,
         iou_threshold=iou_threshold)
