from tqdm import tqdm
from argparse import ArgumentParser
import json
import pandas as pd
import evaluate
from models import Extractor, LoRAExtractor

CONTEXT_TEMPLATE = 'The robot asks the user this question: {} The user responds to that question as follows: {}'

parser = ArgumentParser()
parser.add_argument('-m', '--model_name_or_path', type=str)
parser.add_argument('-d', '--data_path', type=str)
parser.add_argument('-e', '--entities_to_extract', type=lambda s: [str(item) for item in s.split(',')])
parser.add_argument('--lora', action='store_true')
args = parser.parse_args()

extractor = Extractor(name_or_path=args.model_name_or_path) if not args.lora else LoRAExtractor(name_or_path=args.model_name_or_path)
data = pd.read_json(args.data_path)
data['context'] = [CONTEXT_TEMPLATE.format(haru_sent, user_sent) for haru_sent, user_sent in zip(data.haru_sentence, data.user_sentence)]
entities = args.entities_to_extract

predictions = []
for context in tqdm(data['context']):
    extraction_res = extractor.extract(entities=entities, context=context)
    predictions.append(extraction_res)


metric = evaluate.load('squad_v2')
references_for_metric = [dict(id=str(id), answers=ans) for id, ans in zip(data['id'], data['answers'])]
predictions_for_metric = [{'id': str(id), 'prediction_text': pred[entities[0]][0]['text'], 'no_answer_probability': 0.0} for id, pred in zip(data['id'], predictions)]
scores = metric.compute(predictions=predictions_for_metric, references=references_for_metric)
print(scores)

with open('res.json', 'w') as f:
    json.dump(predictions, f, indent='\t')