import uuid

from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response

from chat.gpt.setup_log_analyzer_client import GPTAnomalyAnalyzer
from predictions.predict_results import AnomalyDetectionRobertaModel
from visualization.models import AnomalyFindCounter


class UploaderViewSet(viewsets.ViewSet):
    client = GPTAnomalyAnalyzer()
    anomaly_detect_model_class = AnomalyDetectionRobertaModel()

    @action(detail=False, methods=['post'])
    def find_anomalies(self, request):
        message = request.data.get("message", "")
        log_data = request.data.get("log_data", [])

        if not log_data:
            return Response({"error": "No log data provided."}, status=status.HTTP_400_BAD_REQUEST)

        self.anomaly_detect_model_class.clear_attentions()
        anomaly_logs = []

        for log in log_data:
            predicted_class = self.anomaly_detect_model_class.classify_log(log)
            if predicted_class == 1:
                print(f"Anomaly detected {log}")
                anomaly_logs.append(log)
            else:
                print(f"Not an Anomaly {log} {predicted_class}")

        # if len(anomaly_logs) == 0:
        #     anomaly_logs.append(log_data[0])

        gpt_response = self.client.get_gpt_response(anomaly_logs)

        conversation_history = request.session.get("chat_history", [])
        conversation_history.append({"role": "user", "content": "\n".join(log_data)})
        conversation_history.append({"role": "bot", "content": "\n".join(gpt_response)})
        request.session["chat_history"] = conversation_history


        if not len(gpt_response):
            gpt_response.append("No anomalies detected...!!!")

        AnomalyFindCounter(name=str(uuid.uuid4()), counter=AnomalyFindCounter.get_global_max_counter_value() or 0 + 1).save()

        return Response({'message': gpt_response, 'logs': anomaly_logs})
