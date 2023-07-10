from typing import Any, Dict, List
import inspect
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from utils import prepare_features, postprocess_predictions

ENTITY_MAPPING = {
    "name": "What is the user's name?",
    "hometown": "What is the user's home city?",
    "fav_continent": "What is the user's favorite continent?",
    "next_travel": "What is the user's next travel country, city, or continent?",
    "home_country": "What is the user's home country name?",
    "family_name": "What last name does the user have?",
    "name_origin": "What country or region does the user's family name come from?",
    "profession": "What is the user's profession title?",
    "fav_animal": "What is the user's favorite type of pet?",
    "pet": "What kind of animal is the user's pet?",
    "parents_names": "What are the user's mother and father names?",
    "parents_professions": "What are the user's parents profession titles?",
    "fav_food": "What is the user's favorite food name?",
    "haru_fav_food": "What is the robot's favorite food name?"
}

class BaseExtractor():

    def prepare_inputs(self, features):
        raise NotImplementedError()
    
    @torch.no_grad()
    def extract(self, entities: List[str], context: str, topk: int =1) -> Dict[str, Any]:
        """
        Extract entities from a context.
        Entities are mapped to a question and fed to a QA model using a mapping.
        """

        questions = [self.entity_mapping.get(ent, f"What is the user's {ent}?") for ent in entities]
        examples = [dict(id=f"id_{i}", context=context, question=q) for i,q in enumerate(questions)]
        features = prepare_features(examples=examples, tokenizer=self.tokenizer)
        inputs = self.prepare_inputs(features=features)
        features = [{k: v[i] for k,v in features.items()} for i in range(len(features.example_id))]

        outputs = self.model(**{k: v.to(self.model.device) for k,v in inputs.items()})
        start_end_logits = (outputs.start_logits.cpu().numpy(), outputs.end_logits.cpu().numpy())
        predictions = postprocess_predictions(examples=examples, features=features, predictions=start_end_logits, version_2_with_negative=True)

        results = {
            entity: predictions.popitem(last=False)[1][:topk]
            for entity in entities
        }

        return results


class Extractor(BaseExtractor):

    def __init__(self, name_or_path: str) -> None:
        self.model = AutoModelForQuestionAnswering.from_pretrained(name_or_path, device_map='auto').eval()
        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path)
        self.entity_mapping = ENTITY_MAPPING

    def prepare_inputs(self, features):
        return {k: torch.as_tensor(v) for k,v in features.items() if k in inspect.signature(self.model.forward).parameters.keys()}


class LoRAExtractor(BaseExtractor):

    def __init__(self, name_or_path: str) -> None:
        peft_config = PeftConfig.from_pretrained(name_or_path)
        self.model = AutoModelForQuestionAnswering.from_pretrained(peft_config.base_model_name_or_path, device_map='auto').eval()
        self.model = PeftModel.from_pretrained(model=self.model, model_id=name_or_path, adapter_name='extraction')
        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path)
        self.entity_mapping = ENTITY_MAPPING

    def prepare_inputs(self, features):
        return {k: torch.as_tensor(v) for k,v in features.items() if k in inspect.signature(self.model.get_base_model().forward).parameters.keys()}
