import uuid

from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response

from chat.gpt.setup_log_analyzer_client import GPTAnomalyAnalyzer
from anomaly_detecter_model.anomaly_detection_roberta_model import AnomalyDetectionRobertaModel
from uploader.models import UploadLog
from visualization.models import AnomalyFinderId


class UploaderViewSet(viewsets.ViewSet):
    client = GPTAnomalyAnalyzer()
    anomaly_detect_model_class = AnomalyDetectionRobertaModel()

    @action(detail=False, methods=['post'])
    def find_anomalies(self, request):
        message = request.data.get("message", "")
        log_data = request.data.get("log_data", [])
        file = request.data.get("file", "")

        if not log_data:
            return Response({"error": "No log data provided."}, status=status.HTTP_400_BAD_REQUEST)

        anomaly_finder = AnomalyFinderId(uid=str(uuid.uuid4()), user=str(request.user.id)).save()
        request.session["anomaly_finder_id"] = anomaly_finder.uid

        anomaly_logs = []
        pred_classes = []
        gpt_response = ""

        for log in log_data:
            predicted_class = self.anomaly_detect_model_class.classify_log(log, anomaly_finder.uid)
            pred_classes.append(predicted_class)
            if predicted_class == 1:
                print(f"Anomaly detected {log}")
                anomaly_logs.append(log)
            else:
                print(f"Not an Anomaly {log} {predicted_class}")


            gpt_response = self.client.get_gpt_response(anomaly_logs)

            conversation_history = request.session.get("chat_history", [])
            conversation_history.append({"role": "user", "content": "\n".join(log_data)})
            conversation_history.append({"role": "bot", "content": "\n".join(gpt_response)})
            request.session["chat_history"] = conversation_history


            if not len(gpt_response):
                gpt_response.append("No anomalies detected...!!!")

        logs = UploadLog(file_name=file, logs=log_data, predicted_class= pred_classes, anomaly_finder_id=anomaly_finder.uid)
        logs.save()

        return Response({'message': gpt_response, 'logs': anomaly_logs})

