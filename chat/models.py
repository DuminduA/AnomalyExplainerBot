from datetime import datetime
from mongoengine import Document, StringField, DateTimeField

# TODO save the chat history as well.
class ChatHistory(Document):
    user = StringField()
    date = DateTimeField(default=datetime.now)
    anomaly_finder_id = StringField()