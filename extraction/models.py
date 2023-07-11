from typing import Any, Dict, List
import inspect
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from peft import PeftModel, PeftConfig
from utils import prepare_features, postprocess_predictions

ENTITY_MAPPING = {
    "name": "What is the name of the user?",
    "hometown": "What city is the user from?",
    "fav_continent": "What is the favorite continent of the user?",
    "next_travel": "What is the next travel country, city, or continent of the user?",
    "home_country": "What is the home country name of the user?",
    "family_name": "What is the family name of the user?",
    "name_origin": "What country or region does the user's family name come from?",
    "profession": "What is the user working as?",
    "fav_animal": "What is the favorite animal of the user?",
    "pet": "What kind of animal is the user's pet?",
    "parents_names": "What are the names of the mother and the father of the user?",
    "parents_professions": "What are the user's mother and the father working as?",
    "fav_food": "What is the name of the favorite food of the user?",
    "haru_fav_food": "What is the name of the favorite food of the robot?"
}

class BaseExtractor():

    def prepare_inputs(self, features):
        raise NotImplementedError()
    
    @torch.no_grad()
    def extract(self, context: str, entities: List[str], topk: int =1) -> Dict[str, Any]:
        """
        Extract entities from a context.
        Entities are mapped to a question and fed to a QA model using a mapping.
        """
        questions = [self.entity_mapping.get(ent, f"What is the {ent} of the user?") for ent in entities]
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

    def __init__(self, name_or_path: str, device: str = 'cuda') -> None:
        self.model = AutoModelForQuestionAnswering.from_pretrained(name_or_path).eval().to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path)
        self.entity_mapping = ENTITY_MAPPING

    def prepare_inputs(self, features):
        return {k: torch.as_tensor(v) for k,v in features.items() if k in inspect.signature(self.model.forward).parameters.keys()}


class LoRAExtractor(BaseExtractor):

    def __init__(self, name_or_path: str, device: str = 'cuda') -> None:
        peft_config = PeftConfig.from_pretrained(name_or_path)
        self.model = AutoModelForQuestionAnswering.from_pretrained(peft_config.base_model_name_or_path).eval()
        self.model = PeftModel.from_pretrained(model=self.model, model_id=name_or_path, adapter_name='extraction').to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path)
        self.entity_mapping = ENTITY_MAPPING

    def prepare_inputs(self, features):
        return {k: torch.as_tensor(v) for k,v in features.items() if k in inspect.signature(self.model.get_base_model().forward).parameters.keys()}
