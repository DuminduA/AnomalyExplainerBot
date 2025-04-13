from datetime import datetime

from mongoengine import Document, StringField, ListField, DateTimeField, IntField


class UploadLog(Document):
    file_name = StringField()
    logs = ListField(StringField())
    predicted_class = ListField(IntField())
    date = DateTimeField(default=datetime.now)

