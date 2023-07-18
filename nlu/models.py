from typing import Any, Dict, List, Optional, Tuple, Union

import inspect
import torch
from torch.nn import MSELoss, BCEWithLogitsLoss, CrossEntropyLoss
from transformers import RobertaPreTrainedModel, RobertaModel, RobertaTokenizerFast
from transformers.configuration_utils import PretrainedConfig
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead
from transformers.modeling_outputs import SequenceClassifierOutput, QuestionAnsweringModelOutput
from peft import PeftConfig, PeftModel

from utils import (
    prepare_classification_features, postprocess_classification_predictions,
    prepare_extraction_features, postprocess_extraction_predictions
)


class RobertaForNLU(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler", r"lm_head"] 
    _keys_to_ignore_on_load_missing = [r"position_ids", r"qa_outputs", r"classifier"]

    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config)
        self.qa_outputs = torch.nn.Linear(config.hidden_size, 2)

        # Initialize weights and apply final processing
        self.post_init()

    def classification_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def extraction_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class PeftForNLU(PeftModel):

    def forward(self, *args: Any, **kwargs: Any):
        """
        Forward pass of the model.
        """
        active_adapter = self.active_adapter

        if active_adapter == 'classification':
            return self.get_base_model().classification_forward(*args, **kwargs)
    
        if active_adapter == 'extraction':
            return self.get_base_model().extraction_forward(*args, **kwargs)
        
        return None


class UniversalModel():

    def __init__(self, classifier_path: str, extractor_path: str, device: str = 'cuda') -> None:
        device = device if torch.cuda.is_available() else 'cpu'
        torch_dtype = torch.float16 if device != 'cpu' else torch.float32

        peft_config = PeftConfig.from_pretrained(classifier_path)
        self.model = RobertaForNLU.from_pretrained(peft_config.base_model_name_or_path, num_labels=3, torch_dtype=torch_dtype).eval()
        self.model = PeftForNLU.from_pretrained(model=self.model, model_id=classifier_path, adapter_name='classification').to(device)
        self.model.load_adapter(model_id=extractor_path, adapter_name='extraction')
        self.tokenizer = RobertaTokenizerFast.from_pretrained(classifier_path)

    @torch.no_grad()
    def classify(self, document: str, queries: List[str], labels: List[str], no_class: bool = False) -> Dict[str, Any]:
        """
        Classify the document into one of the labels.
        """
        self.model.set_adapter('classification')

        threshold = 2 / (len(labels) + 1)
        features = prepare_classification_features(document=document, candidates=queries, tokenizer=self.tokenizer)
        inputs = {k: torch.as_tensor(v) for k,v in features.items() if k in inspect.signature(self.model.get_base_model().classification_forward).parameters.keys()}

        outputs = self.model(**{k: v.to(self.model.device) for k,v in inputs.items()})
        predictions = postprocess_classification_predictions(predictions=outputs.logits.cpu().numpy())

        results = dict(
            label='' if no_class and (predictions['prediction']['score'] < threshold) else labels[predictions['prediction']['idx']],
            score=float(predictions['prediction']['score'])
        )

        return results
    
    @torch.no_grad()
    def extract(self, document: str, queries: List[str], entities: List[str], topk: int = 1) -> Dict[str, Any]:
        """
        Extract entities from a context.
        Entities are mapped to a question and fed to a QA model using a mapping.
        """
        self.model.set_adapter('extraction')

        examples = [dict(id=f"id_{i}", context=document, question=q) for i,q in enumerate(queries)]
        features = prepare_extraction_features(examples=examples, tokenizer=self.tokenizer)
        inputs = {k: torch.as_tensor(v) for k,v in features.items() if k in inspect.signature(self.model.get_base_model().extraction_forward).parameters.keys()}
        features = [{k: v[i] for k,v in features.items()} for i in range(len(features.example_id))]

        outputs = self.model(**{k: v.to(self.model.device) for k,v in inputs.items()})
        start_end_logits = (outputs.start_logits.cpu().numpy(), outputs.end_logits.cpu().numpy())
        predictions = postprocess_extraction_predictions(examples=examples, features=features, predictions=start_end_logits, version_2_with_negative=True)

        results = {
            entity: predictions.popitem(last=False)[1][:topk]
            for entity in entities
        }

        return results
