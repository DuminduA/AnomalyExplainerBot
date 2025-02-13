from django.shortcuts import render
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response

from chat.gpt.setup_log_analyzer_client import GPTAnomalyAnalyzer
from predictions.predict_results import AnomalyDetectionRobertaModel


def upload(request):
    return render(request, 'uploader/uploader.html')


class UploaderViewSet(viewsets.ViewSet):
    client = GPTAnomalyAnalyzer()
    model = AnomalyDetectionRobertaModel()

    @action(detail=False, methods=['post'])
    def find_anomalies(self, request):
        message = request.data.get("message", "")
        log_data = request.data.get("log_data", [])

        if not log_data:
            return Response({"error": "No log data provided."}, status=status.HTTP_400_BAD_REQUEST)

        anomaly_logs = []

        for log in log_data:
            predicted_class = self.model.classify_log(log)
            if predicted_class:
                anomaly_logs.append(log)

        gpt_response = self.client.get_gpt_response(anomaly_logs)


        return render(request, 'chat/home.html', {'message': gpt_response, 'logs': anomaly_logs})
