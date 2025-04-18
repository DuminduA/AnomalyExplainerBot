import os
import torch
import logging
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
        self.attentions = []

    def load_model(self):
        # login(settings.HUGGING_FACE_WRITE_API_KEY)

        model = RobertaForSequenceClassification.from_pretrained("Dumi2025/log-anomaly-detection-model", token=settings.HUGGING_FACE_WRITE_API_KEY)
        tokenizer = RobertaTokenizerFast.from_pretrained("Dumi2025/log-anomaly-detection-model", token=settings.HUGGING_FACE_WRITE_API_KEY)

        # base_llama_model = AutoModelForSequenceClassification.from_pretrained(
        #     "meta-llama/Llama-2-7b-hf",
        #     device_map="auto"
        # )
        # model = PeftModel.from_pretrained(base_llama_model, "Dumi2025/log-anomaly-detection-model-llama", token=settings.HUGGING_FACE_WRITE_API_KEY)
        # tokenizer = LlamaTokenizer.from_pretrained("Dumi2025/log-anomaly-detection-model-llama", token=settings.HUGGING_FACE_WRITE_API_KEY)

        model.to("cpu")
        model.eval()

        return model, tokenizer


    def classify_log(self, log):
        inputs = self.tokenizer(log, return_tensors="pt", truncation=True, padding=True)

        # Perform classification
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=-1).item()
            attentions = outputs.attentions
            self.attentions.append({"inputs": inputs, "attentions": attentions})

        return predicted_class

    def clear_attentions(self):
        self.attentions = []

