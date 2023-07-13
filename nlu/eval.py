from tqdm import tqdm
from argparse import ArgumentParser
import json
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import evaluate

from models import UniversalModel
from tasks import IntentRecognition, BoolQA, SentimentAnalysis, NamedEntityRecognition

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

INTENT_MAPPING = {
    'topic-intro': ['name'],
    'topic-hometown-what-question': ['hometown'],
    'favorite-continent-question': ['fav_continent'],
    'where-do-you-want-to-travel-next-question': ['next_travel'],
    'topic-travel-homecountry-what-is-your-home-country-question': ['home_country'],
    'speakers-last-name-question': ['family_name'],
    'speakers-tells-lastname': ['name_origin'],
    'topic-profession': ['profession'],
    'topic-pet-generic-followup': ['fav_animal'],
    'topic-pet': ['pet'],
    'topic-day-two-parents': ['parents_names'],
    'topic-day-two-parents-occupation': ['parents_professions'],
    'topic-day-three-food': ['fav_food'],
    'topic-day-three-haru-food': ['haru_fav_food']
}
TASK_MAPPING = {'intent': IntentRecognition(), 'boolqa': BoolQA(), 'sentiment': SentimentAnalysis(), 'ner': NamedEntityRecognition()}

parser = ArgumentParser()
parser.add_argument('--classifier_path', type=str)
parser.add_argument('--extractor_path', type=str)
parser.add_argument('-d', '--data_path', type=str)
parser.add_argument('--lora', action='store_true')
parser.add_argument('-t', '--task', choices=['intent', 'boolqa', 'sentiment', 'ner'])
parser.add_argument('--no_class', action='store_true')
args = parser.parse_args()

classifier = UniversalModel(classifier_path=args.classifier_path, extractor_path=args.extractor_path)

if args.task == 'ner':
    data = pd.read_json(args.data_path, dtype={'id': str}).set_index('id')
    data['entities_to_extract'] = [INTENT_MAPPING[intent] for intent in data.intent]
else:
    data = pd.read_json(args.data_path)

predictions = []
pbar = tqdm(range(len(data)), leave=False, desc='Evaluation')
for id, sample in data.iterrows():
    pbar.update()
    
    task = TASK_MAPPING[args.task]
    inputs = task.generate_inputs(**sample.to_dict())

    if args.task == 'ner':
        res = classifier.extract(**inputs, entities=sample.entities_to_extract)
        predictions.append(dict(id=id, entities=sample.entities_to_extract, sentence=sample.user_sentence, prediction=res))
    else:
        res = classifier.classify(**inputs, labels=sample.candidate_labels, no_class=args.no_class)
        predictions.append(res)


if args.task == 'ner':
    metric = evaluate.load('squad_v2')
    references_for_metric = [dict(id=id, answers=sample.answers) for id, sample in data.iterrows()]
    predictions_for_metric = [{'id': pred['id'], 'prediction_text': pred['prediction'][pred['entities'][0]][0]['text'], 'no_answer_probability': 0.0} for pred in predictions]
    scores = metric.compute(predictions=predictions_for_metric, references=references_for_metric)
    print(scores)
else:
    predictions_for_metric = [x['label'] for x in predictions]
    references_for_metric = [l if l is not None else '' for l in data.label]
    acc = accuracy_score(y_true=references_for_metric, y_pred=predictions_for_metric)
    prec, recall, f1, _ = precision_recall_fscore_support(y_true=references_for_metric, y_pred=predictions_for_metric, average='weighted')
    print('Accuracy: {}'.format(acc))
    print('Precision: {}'.format(prec))
    print('Recall: {}'.format(recall))
    print('F1: {}'.format(f1))

    # for i, (idx, sample) in enumerate(data.iterrows()):
    #     if predictions_for_metric[i] != references_for_metric[i]:
    #         print(bcolors.FAIL)
    #         print(sample.user_sentence, predictions[i], sample.label)
    #     else:
    #         print(bcolors.OKBLUE)
    #         print(sample.user_sentence, predictions[i], sample.label)
    #     print(bcolors.ENDC)

with open('res.json', 'w') as f:
    json.dump(predictions, f, indent='\t')

