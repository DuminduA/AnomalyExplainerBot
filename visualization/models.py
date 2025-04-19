from datetime import datetime
from mongoengine import Document, ListField, StringField, IntField, DateTimeField, DynamicField


class AnomalyFinderId(Document):
    uid = StringField(unique=True)
    created_at = DateTimeField(default=datetime.now())
    user = StringField()

class AttentionData(Document):
    tokens = ListField(StringField())
    attn = DynamicField()
    date = DateTimeField(default=datetime.now)
    anomaly_finder_id = StringField()


