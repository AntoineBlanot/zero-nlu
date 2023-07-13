from typing import Any, Dict, List
import inspect
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel, PeftConfig
from utils import prepare_features, postprocess_predictions

class BaseClassifier():

    def prepare_inputs(self, features):
        raise NotImplementedError()
    
    @torch.no_grad()
    def classify(self, document: str, candidates: List[str], labels: List[str], no_class: bool = False) -> Dict[str, Any]:
        """
        Classify the document into one of the labels.
        """
        threshold = 2 / (len(labels) + 1)
        features = prepare_features(document=document, candidates=candidates, tokenizer=self.tokenizer)
        inputs = self.prepare_inputs(features=features)

        outputs = self.model(**{k: v.to(self.model.device) for k,v in inputs.items()})
        predictions = postprocess_predictions(predictions=outputs.logits.cpu().numpy())

        results = dict(
            label='' if no_class and (predictions['prediction']['score'] < threshold) else labels[predictions['prediction']['idx']],
            score=float(predictions['prediction']['score'])
        )

        return results


class Classifier(BaseClassifier):

    def __init__(self, name_or_path: str, device: str = 'cuda') -> None:
        self.model = AutoModelForSequenceClassification.from_pretrained(name_or_path, num_labels=3).eval().to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path)

    def prepare_inputs(self, features):
        return {k: torch.as_tensor(v) for k,v in features.items() if k in inspect.signature(self.model.forward).parameters.keys()}


class LoRAClassifier(BaseClassifier):

    def __init__(self, name_or_path: str, device: str = 'cuda') -> None:
        peft_config = PeftConfig.from_pretrained(name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(peft_config.base_model_name_or_path, num_labels=3).eval()
        self.model = PeftModel.from_pretrained(model=self.model, model_id=name_or_path, adapter_name='classification').to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path)

    def prepare_inputs(self, features):
        return {k: torch.as_tensor(v) for k,v in features.items() if k in inspect.signature(self.model.get_base_model().forward).parameters.keys()}
