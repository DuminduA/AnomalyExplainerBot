import os
import torch
import logging

from django.conf import settings
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification

from visualization.models import ModelAttentions

# Disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.device("cpu")

logger = logging.getLogger(__name__)


class AnomalyDetectionRobertaModel:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AnomalyDetectionRobertaModel, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.repository_id = "Dumi2025/log-anomaly-detection-model-new"
        self.model = self._load_model()
        self.tokenizer = self._load_tokenizer()
        self._initialized = True

    def _load_model(self):
        model = RobertaForSequenceClassification.from_pretrained(
            self.repository_id,
            token=settings.HUGGING_FACE_WRITE_API_KEY
        )
        model.to("cpu")
        model.eval()
        return model

    def _load_tokenizer(self):
        return RobertaTokenizerFast.from_pretrained(
            self.repository_id,
            token=settings.HUGGING_FACE_WRITE_API_KEY
        )

    def classify(self, log_text: str) -> int:
        inputs = self.tokenizer(log_text, return_tensors="pt", truncation=True, padding=True)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            return torch.argmax(logits, dim=-1).item()

    def classify_with_attention(self, log_text: str, anomaly_finder_id: int) -> int:
        inputs = self.tokenizer(log_text, return_tensors="pt", truncation=True, padding=True)

        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
            predicted_class = torch.argmax(outputs.logits, dim=-1).item()
            attentions = outputs.attentions
            input_ids = inputs["input_ids"].squeeze().tolist()

        self._save_attention(log_text, input_ids, attentions, anomaly_finder_id)
        return predicted_class

    def _save_attention(self, log_text, input_ids, attentions, anomaly_finder_id):
        attentions_as_lists = [layer.tolist() for layer in attentions]
        attention_record = ModelAttentions(
            attentions=attentions_as_lists,
            input_ids=input_ids,
            log=log_text,
            anomaly_finder_id=anomaly_finder_id
        )
        attention_record.save()