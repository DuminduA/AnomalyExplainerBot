from datetime import datetime
from mongoengine import Document, ListField, StringField, IntField, DateTimeField, DynamicField


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
    attentions = ListField()
    input_ids = ListField()
    log = StringField()
    anomaly_finder_id = StringField()
    created_at = DateTimeField(default=datetime.now())




