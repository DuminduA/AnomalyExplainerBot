from anomaly_detecter_model.anomaly_detection_roberta_model import AnomalyDetectionRobertaModel
from chat.gpt.setup_log_analyzer_client import GPTAnomalyAnalyzer
from uploader.models import UploadLog

class FileUploaderService:
    anomaly_detect_model_class = AnomalyDetectionRobertaModel()
    client = GPTAnomalyAnalyzer()

    def get_anomalies(self, log_data, anomaly_finder):
        anomaly_logs = []
        predicted_classes = []

        for log in log_data:
            predicted_class = self.anomaly_detect_model_class.classify_with_attention(log, anomaly_finder.uid)
            predicted_classes.append(predicted_class)

            if predicted_class == 1:
                print(f"Anomaly detected: {log}")
                anomaly_logs.append(log)
            else:
                print(f"Not an anomaly: {log} (Class {predicted_class})")
        return anomaly_logs, predicted_classes

    def create_anomaly_bot_message(self, anomaly_logs, conversation_history, log_data):
        if anomaly_logs:
            gpt_response = self.client.get_gpt_response(anomaly_logs)
        else:
            gpt_response = ["No anomalies detected...!!!"]

        # Update conversation history
        conversation_history.append({"role": "user", "content": "\n".join(log_data)})
        conversation_history.append({"role": "bot", "content": "\n".join(gpt_response)})

        return conversation_history, gpt_response

    def save_logs(self, file, log_data, predicted_classes, anomaly_finder):
        logs = UploadLog(
            file_name=file,
            logs=log_data,
            predicted_class=predicted_classes,
            anomaly_finder_id=anomaly_finder.uid
        )
        logs.save()

    def process_file(self, conversation_history, file, log_data, anomaly_finder):
        anomaly_logs, predicted_classes = self.get_anomalies(log_data, anomaly_finder)
        conversation_history, gpt_response = self.create_anomaly_bot_message(anomaly_logs, conversation_history, log_data)
        self.save_logs(file, log_data, predicted_classes, anomaly_finder)

        return {
            "gpt_response": gpt_response,
            "conversation_history": conversation_history,
            "anomaly_logs": anomaly_logs
        }
