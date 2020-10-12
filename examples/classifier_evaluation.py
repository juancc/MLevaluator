from  MLinference.architectures.KerasClassifiers import KerasClassifiers

from MLevaluator.evaluate import evaluate

dataset = '/misdoc/datasets/baluarte/00025/annotation.json'
labels = ['con arnes','sin arnes']
save_path = '/home/juanc/tmp/model_evaluation/arnes'
model_path = '/home/juanc/Downloads/resnet_imageAI_arnes_v1.ml'

model = KerasClassifiers.load(model_path)

evaluate(model, dataset,
         save_path=save_path,
         parents=['persona'],
         labels=labels,
         percentage=0.01,
         debug=False)