import os
import torch
import logging

from visualization.models import ModelAttentions

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from django.conf import settings
from huggingface_hub import login
from transformers import RobertaTokenizerFast,RobertaForSequenceClassification, AutoModelForSequenceClassification, AutoTokenizer, LlamaForSequenceClassification, LlamaTokenizer
from peft import PeftModel

torch.device("cpu")

logger = logging.getLogger(__name__)

class AnomalyDetectionRobertaModel:
    def __init__(self):
        self.model, self.tokenizer = self.load_model()

    def load_model(self):
        repository_id = "Dumi2025/log-anomaly-detection-model-new"
        model = RobertaForSequenceClassification.from_pretrained(repository_id, token=settings.HUGGING_FACE_WRITE_API_KEY)
        tokenizer = RobertaTokenizerFast.from_pretrained(repository_id, token=settings.HUGGING_FACE_WRITE_API_KEY)

        model.to("cpu")
        model.eval()

        return model, tokenizer


    def classify_log(self, log, anomaly_finder_id):
        inputs = self.tokenizer(log, return_tensors="pt", truncation=True, padding=True)

        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=-1).item()
            attentions = outputs.attentions  # Tuple of tensors

            input_ids = inputs["input_ids"].squeeze().tolist()

            attentions_as_lists = [layer[0].tolist() for layer in attentions]

            model_attention = ModelAttentions(
                attentions=attentions_as_lists,
                input_ids=input_ids,
                log=log,
                anomaly_finder_id=anomaly_finder_id
            )
            model_attention.save()

        return predicted_class

