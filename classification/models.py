from typing import Any, Dict, List
import inspect
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel, PeftConfig
from utils import prepare_features, postprocess_predictions

LABEL_MAPPING = {
    "watch-in-person": "I like to watch where the action is.",
    "watches-on-tv": "I like to watch on the television.",

    "natural-wonders": "I like nature.",
    "man-made-monuments-answer": "I like monuments.",

    "topic-books-physical-books": "I like physical books.",
    "topic-books-ebooks": "I like electronic books.",

    "topic-books-most-sold-book-rowling": "It is J. K. Rowling.",
    "topic-books-most-sold-book-tolkien": "It is J. R. R. Tolkien.",

    "topic-food-for-breakfast": "It is for breakfast.",
    "topic-food-for-lunch": "It is for lunch.",
    "topic-food-for-dinner": "It is for dinner.",

    "topic-hometown-type-of-building-apartment-answer": "I live in an apartment.",
    "topic-hometown-type-of-building-house-answer": "I live in a house.",

    # "topic-pet-eat-answer-little": "my pet eats a little",
    # "topic-pet-eat-answer-lots": "my pet eats a lot",

    "topic-speaker-age-less-than-18-answer": "I am less than 18 years old.",
    "topic-speaker-age-greater-than-18-answer": "I am more than 18 years old.",

    "topic-travel-homecountry-favorite-hemisphere-north": "I like the North.",
    "topic-travel-homecountry-favorite-hemisphere-south": "I like the South.",

    "positive": "This text expresses a positive sentiment.",
    "negative": "This text expresses a negative sentiment.",
    "neutral": "This text expresses a neutral sentiment."
}

class BaseClassifier():

    def prepare_inputs(self, features):
        raise NotImplementedError()
    
    @torch.no_grad()
    def classify(self, document: str, labels: List[str], no_class: bool = False) -> Dict[str, Any]:
        """
        Classify the document into one of the labels.
        """
        threshold = 2 / (len(labels) + 1)
        candidates = [self.label_mapping.get(label, f"The document is about {label}.") for label in labels]
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
        self.label_mapping = LABEL_MAPPING

    def prepare_inputs(self, features):
        return {k: torch.as_tensor(v) for k,v in features.items() if k in inspect.signature(self.model.forward).parameters.keys()}


class LoRAClassifier(BaseClassifier):

    def __init__(self, name_or_path: str, device: str = 'cuda') -> None:
        peft_config = PeftConfig.from_pretrained(name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(peft_config.base_model_name_or_path, num_labels=3).eval()
        self.model = PeftModel.from_pretrained(model=self.model, model_id=name_or_path, adapter_name='extraction').to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path)
        self.label_mapping = LABEL_MAPPING

    def prepare_inputs(self, features):
        return {k: torch.as_tensor(v) for k,v in features.items() if k in inspect.signature(self.model.get_base_model().forward).parameters.keys()}
