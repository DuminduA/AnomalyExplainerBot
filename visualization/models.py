from datetime import datetime
from mongoengine import Document, ListField, StringField, IntField, DateTimeField, DynamicField


class AnomalyFindCounter(Document):
    name = StringField(unique=True)
    counter = IntField(default=0)

    @staticmethod
    def get_global_max_counter_value():
        counter = AnomalyFindCounter.objects.order_by('-counter').first()
        if counter:
            return counter.counter
        return None

class AttentionData(Document):
    tokens = ListField(StringField())
    attn = DynamicField()
    date = DateTimeField(default=datetime.now)
    custom_id = IntField(unique=True)

    def save(self, *args, **kwargs):
        if not self.custom_id:
            counter = AnomalyFindCounter.get_global_max_counter_value()
            if not counter:
                counter_obj = AnomalyFindCounter(name='attention_data_counter', counter=0)
                counter_obj.save()
                counter = counter_obj.counter

            self.custom_id = counter
        return super().save(*args, **kwargs)


