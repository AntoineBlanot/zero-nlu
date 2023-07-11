from tqdm import tqdm
from argparse import ArgumentParser
import json
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from models import Classifier, LoRAClassifier

TASK = 'intent'
DOCUMENT_TEMPLATE = '{} {}'

parser = ArgumentParser()
parser.add_argument('-m', '--model_name_or_path', type=str)
parser.add_argument('-d', '--data_path', type=str)
parser.add_argument('--lora', action='store_true')
args = parser.parse_args()

classifier = Classifier(name_or_path=args.model_name_or_path) if not args.lora else LoRAClassifier(name_or_path=args.model_name_or_path)
data = pd.read_json(args.data_path)
# data = data.loc[data.intent != 'topic-pet-eat-question']

predictions = []
pbar = tqdm(range(len(data)), leave=False, desc='Evaluation')
for i, sample in data.iterrows():
    pbar.update()
    
    if TASK in ['intent', 'sentiment']:
        document = sample.user_sentence
    elif TASK == 'boolqa':
        document = DOCUMENT_TEMPLATE.format(sample.haru_sentence, sample.user_sentence)
    
    extraction_res = classifier.classify(document=document, labels=sample.candidate_labels, no_class=True)
    predictions.append(extraction_res)

predictions_for_metrics = [x['label'] for x in predictions]
references_for_metrics = [l if l is not None else '' for l in data.label]
acc = accuracy_score(y_true=references_for_metrics, y_pred=predictions_for_metrics)
prec, recall, f1, _ = precision_recall_fscore_support(y_true=references_for_metrics, y_pred=predictions_for_metrics, average='weighted')
print('Accuracy: {}'.format(acc))
print('Precision: {}'.format(prec))
print('Recall: {}'.format(recall))
print('F1: {}'.format(f1))

# for i, (idx, sample) in enumerate(data.iterrows()):
#     print(sample.user_sentence, predictions[i], sample.label)

with open('res.json', 'w') as f:
    json.dump(predictions, f, indent='\t')

