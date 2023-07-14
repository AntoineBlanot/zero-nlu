import unittest
import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel, PeftConfig

class TestClassification(unittest.TestCase):

    def setUp(self) -> None:
        name_or_path = os.environ['MODEL_PATH']
        peft_config = PeftConfig.from_pretrained(name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(peft_config.base_model_name_or_path, num_labels=3).eval()
        self.model = PeftModel.from_pretrained(model=self.model, model_id=name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path)

    def test_peft_load(self):
        base_classifier_tensor = self.model.base_model.model.classifier.original_module.dense.weight
        peft_classifier_tensor = self.model.base_model.model.classifier.modules_to_save.default.dense.weight
        self.assertFalse(torch.equal(base_classifier_tensor, peft_classifier_tensor))

    def test_forward(self):
        premise = 'I like football.'
        hypothesis = 'I enjoy football.'
        expected_logits = torch.as_tensor([[ 3.8265, -0.4996, -3.1603]])

        inputs = self.tokenizer(premise, hypothesis, return_tensors='pt')
        outputs = self.model(**inputs)

        logits = outputs.logits.cpu()
        self.assertTrue(torch.allclose(logits, expected_logits, rtol=1e-3))

if __name__ == '__main__':
    unittest.main()