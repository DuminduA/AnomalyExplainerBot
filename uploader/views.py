import uuid

from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response

from explainer.bertviz_service import process_model_attentions
from visualization.models import AnomalyFinderId, ModelAttentions
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

        anomaly_finder = self.create_new_anomaly_finder(request.user.id, request.session.session_key)
        request.session["anomaly_finder_id"] = anomaly_finder.uid

        results_dict = self.service.process_file(conversation_history, file, log_data, anomaly_finder)
        self.generate_visualization_data(anomaly_finder.uid)
        request.session["chat_history"] = results_dict.get("conversation_history")

        return Response({'message': results_dict.get("gpt_response"), 'logs': results_dict.get("anomaly_logs")})

    def create_new_anomaly_finder(self, user_id, session_id):
        anomaly_finder = AnomalyFinderId(uid=str(uuid.uuid4()), user=str(user_id), session_id=session_id).save()
        return anomaly_finder

    def create_new_session(self, request):
        request.session.flush()

    def generate_visualization_data(self, anomaly_finder_id):
        model_attentions = ModelAttentions.objects.filter(anomaly_finder_id=anomaly_finder_id)
        if not model_attentions:
            return
        process_model_attentions(model_attentions, anomaly_finder_id,[],[],[], True)

