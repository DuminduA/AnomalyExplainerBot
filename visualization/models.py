from datetime import datetime

import torch
from mongoengine import Document, ListField, StringField, IntField, DateTimeField, DynamicField, FloatField


class AnomalyFinderId(Document):
    uid = StringField(unique=True)
    created_at = DateTimeField(default=datetime.now())
    user = StringField()


class BertvizAttentionData(Document):
    tokens = ListField(StringField())
    attn = DynamicField()
    date = DateTimeField(default=datetime.now())
    anomaly_finder_id = StringField()


class CaptumAttentionData(Document):
    attributions = ListField()
    delta = ListField()
    created_at = DateTimeField(default=datetime.now())
    anomaly_finder_id = StringField()


class ModelAttentions(Document):
    attentions = ListField(ListField(ListField(ListField(FloatField()))))  # [layer][head][seq_len][seq_len]
    input_ids = ListField(IntField())
    log = StringField()
    anomaly_finder_id = StringField()
    created_at = DateTimeField(default=datetime.now)

    def get_attention_tensors(self):
        """
        Convert nested lists back to tuple of torch.Tensors for visualization with BertViz.
        Returns:
            Tuple[torch.Tensor]: A tuple of attention tensors (one per layer)
        """
        return tuple(torch.tensor(layer) for layer in self.attentions)




