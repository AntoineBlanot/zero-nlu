from typing import List, Dict, Any
import numpy as np
from scipy.special import softmax, expit

def prepare_features(document: str, candidates: List[str], tokenizer) -> List[Dict[str, Any]]:
    document = [document] * len(candidates)

    tokenized_examples = tokenizer(
        document,
        candidates,
        truncation="only_first",
        max_length=128,
        padding=True
    )

    return tokenized_examples

def postprocess_predictions(predictions: np.ndarray, entail_idx: int = 0) -> Dict[str, Any]:
    """
    Post-processes the predictions of a text-classification model to convert them to zero-shot classification outputs.

    Args:
        predictions (:obj:`np.ndarray`):
            The predictions of the model.
        entail_idx (:obj:`int`):
            Index of the entailment class (for models trained on NLI tasks). Default to value of https://huggingface.co/datasets/multi_nli
    """
    outputs = predictions[:, entail_idx]
    probs = softmax(outputs, axis=-1) if len(outputs) > 1 else expit(outputs)

    predictions = dict(prediction=dict(idx=outputs.argmax(-1), score=probs.max(-1)), scores=probs, logits=outputs)

    return predictions