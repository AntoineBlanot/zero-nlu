import unittest
import os
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from peft import PeftModel, PeftConfig

class TestClassification(unittest.TestCase):

    def setUp(self) -> None:
        name_or_path = os.environ['MODEL_PATH']
        peft_config = PeftConfig.from_pretrained(name_or_path)
        self.model = AutoModelForQuestionAnswering.from_pretrained(peft_config.base_model_name_or_path).eval()
        self.model = PeftModel.from_pretrained(model=self.model, model_id=name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path)

    def test_peft_load(self):
        base_classifier_tensor = self.model.base_model.model.qa_outputs.original_module.weight
        peft_classifier_tensor = self.model.base_model.model.qa_outputs.modules_to_save.default.weight
        self.assertFalse(torch.equal(base_classifier_tensor, peft_classifier_tensor))

    def test_forward(self):
        context = 'My name is HARU.'
        question = 'What is my name?'
        expected_start_logits = torch.as_tensor([[ -0.2834,  -1.1120,  -3.7747,  -6.5744,   2.1181,  -4.2417,  -5.1048,
          -8.7048,  -9.7222,  -9.8521, -12.0208,  -7.8550,  -7.7753,  -9.9638,
          -5.1048]])

        inputs = self.tokenizer(context, question, return_tensors='pt')
        outputs = self.model(**inputs)

        start_logits = outputs.start_logits.cpu()
        print(start_logits)
        self.assertTrue(torch.allclose(start_logits, expected_start_logits, rtol=1e-3))

if __name__ == '__main__':
    unittest.main()