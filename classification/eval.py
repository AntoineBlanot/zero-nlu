from tqdm import tqdm
from argparse import ArgumentParser
import json
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from models import Classifier, LoRAClassifier
from tasks import IntentRecognition, BoolQA, SentimentAnalysis

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

TASK_MAPPING = {'intent': IntentRecognition(), 'boolqa': BoolQA(), 'sentiment': SentimentAnalysis()}

parser = ArgumentParser()
parser.add_argument('-m', '--model_name_or_path', type=str)
parser.add_argument('-d', '--data_path', type=str)
parser.add_argument('--lora', action='store_true')
parser.add_argument('-t', '--task', choices=['intent', 'boolqa', 'sentiment'])
parser.add_argument('--no_class', action='store_true')
args = parser.parse_args()

classifier = Classifier(name_or_path=args.model_name_or_path) if not args.lora else LoRAClassifier(name_or_path=args.model_name_or_path)
data = pd.read_json(args.data_path)
# data = data.loc[data.intent == 'topic-olympics-user-guess-haru-height']

predictions = []
pbar = tqdm(range(len(data)), leave=False, desc='Evaluation')
for i, sample in data.iterrows():
    pbar.update()
    
    task = TASK_MAPPING[args.task]
    document = task.generate_document(document=sample.user_sentence, question=sample.haru_sentence)
    candidates = task.generate_candidates(labels=sample.candidate_labels, question=sample.haru_sentence)

    extraction_res = classifier.classify(document=document, candidates=candidates, labels=sample.candidate_labels, no_class=args.no_class)
    predictions.append(extraction_res)

predictions_for_metric = [x['label'] for x in predictions]
references_for_metric = [l if l is not None else '' for l in data.label]
acc = accuracy_score(y_true=references_for_metric, y_pred=predictions_for_metric)
prec, recall, f1, _ = precision_recall_fscore_support(y_true=references_for_metric, y_pred=predictions_for_metric, average='weighted')
print('Accuracy: {}'.format(acc))
print('Precision: {}'.format(prec))
print('Recall: {}'.format(recall))
print('F1: {}'.format(f1))

for i, (idx, sample) in enumerate(data.iterrows()):
    if predictions_for_metric[i] != references_for_metric[i]:
        print(bcolors.FAIL)
        print(sample.user_sentence, predictions[i], sample.label)
    else:
        print(bcolors.OKBLUE)
        print(sample.user_sentence, predictions[i], sample.label)
    print(bcolors.ENDC)

with open('res.json', 'w') as f:
    json.dump(predictions, f, indent='\t')

