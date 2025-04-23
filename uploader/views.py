import uuid

from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response

from visualization.models import AnomalyFinderId
from uploader.service import FileUploaderService


class UploaderViewSet(viewsets.ViewSet):
    service = FileUploaderService()

    @action(detail=False, methods=['post'])
    def find_anomalies(self, request):
        log_data = request.data.get("log_data", [])

        if not log_data:
            return Response({"error": "No log data provided."}, status=status.HTTP_400_BAD_REQUEST)
        if len(log_data)> 10:
            return Response({"error": "Too many log data uploaded. Please only give files with at most 10 logs."}, status=status.HTTP_400_BAD_REQUEST)

        message = request.data.get("message", "")
        file = request.data.get("file", "")
        conversation_history = request.session.get("chat_history", [])

        anomaly_finder = self.create_new_anomaly_finder(request.user.id)
        request.session["anomaly_finder_id"] = anomaly_finder.uid

        results_dict = self.service.process_file(conversation_history, file, log_data, anomaly_finder)
        request.session["chat_history"] = results_dict.get("conversation_history")

        return Response({'message': results_dict.get("gpt_response"), 'logs': results_dict.get("anomaly_logs")})

    def create_new_anomaly_finder(self, user_id):
        anomaly_finder = AnomalyFinderId(uid=str(uuid.uuid4()), user=str(user_id)).save()
        return anomaly_finder
