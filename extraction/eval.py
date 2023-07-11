from tqdm import tqdm
from argparse import ArgumentParser
import json
import pandas as pd
import evaluate
from models import Extractor, LoRAExtractor

CONTEXT_TEMPLATE = 'The robot asks the user this question: {} The user responds to that question as follows: {}'
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

parser = ArgumentParser()
parser.add_argument('-m', '--model_name_or_path', type=str)
parser.add_argument('-d', '--data_path', type=str)
parser.add_argument('--lora', action='store_true')
args = parser.parse_args()

extractor = Extractor(name_or_path=args.model_name_or_path) if not args.lora else LoRAExtractor(name_or_path=args.model_name_or_path)
data = pd.read_json(args.data_path, dtype={'id': str}).set_index('id')
data['context'] = [CONTEXT_TEMPLATE.format(haru_sent, user_sent) for haru_sent, user_sent in zip(data.haru_sentence, data.user_sentence)]
data['entities'] = [INTENT_MAPPING[intent] for intent in data.intent]

predictions = []
pbar = tqdm(range(len(data)), leave=False, desc='Evaluation')
for id, sample in data.iterrows():
    pbar.update()
    extraction_res = extractor.extract(context=sample.context, entities=sample.entities)
    predictions.append(dict(id=id, entities=sample.entities, sentence=sample.user_sentence, prediction=extraction_res))

metric = evaluate.load('squad_v2')
references_for_metric = [dict(id=id, answers=sample.answers) for id, sample in data.iterrows()]
predictions_for_metric = [{'id': pred['id'], 'prediction_text': pred['prediction'][pred['entities'][0]][0]['text'], 'no_answer_probability': 0.0} for pred in predictions]
scores = metric.compute(predictions=predictions_for_metric, references=references_for_metric)
print(scores)

with open('res.json', 'w') as f:
    json.dump(predictions, f, indent='\t')