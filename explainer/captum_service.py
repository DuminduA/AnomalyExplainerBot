import torch
from captum.attr import IntegratedGradients
from torch import nn
import torch.nn.functional as F
from captum.attr import visualization as viz

from uploader.models import UploadLog
from uploader.views import UploaderViewSet
from visualization.models import CaptumAttentionData

anomaly_detect_model_class = UploaderViewSet.anomaly_detect_model_class

class RobertaWrapper(nn.Module):
    def __init__(self, model):
        super(RobertaWrapper, self).__init__()
        self.model = model

    def forward(self, inputs, attention_mask=None):
        outputs = self.model(inputs, attention_mask=attention_mask)
        return F.softmax(outputs.logits, dim=-1)


# Create wrapped model
wrapped_model = RobertaWrapper(anomaly_detect_model_class.model)

def predict(inputs, attention_mask=None):
    return wrapped_model(inputs.long(), attention_mask)

ig = IntegratedGradients(predict)

def save_feature_attribution_with_ig(request):
    anomaly_finder_id=request.session.get("anomaly_finder_id")
    if not anomaly_finder_id:
        raise ValueError(f"Could not find anomaly finder id {anomaly_finder_id}")
    logs = UploadLog.objects.filter(anomaly_finder_id=anomaly_finder_id).first()
    attributions_list = []
    deltas_list = []

    for log in logs['logs']:
        inputs = anomaly_detect_model_class.tokenizer(log, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].long()
        attention_mask = inputs["attention_mask"]

        attributions, delta = ig.attribute(
            input_ids,
            target=0,
            n_steps=50,
            return_convergence_delta=True,
            additional_forward_args=attention_mask
        )

        attributions = attributions.sum(dim=-1).squeeze(0)
        attributions = attributions / torch.norm(attributions)

        attributions_list.append(attributions)
        deltas_list.append(delta)

    CaptumAttentionData(attributions=attributions_list, delta=deltas_list, anomaly_finder_id=anomaly_finder_id)

# def save_layer_attributions():
#     # Configure interpretable embedding layer
#     interpretable_embedding = configure_interpretable_embedding_layer(model, 'roberta.embeddings')
#
#     # Create layer attribution method
#     lig = LayerIntegratedGradients(predict, interpretable_embedding)
#
#     # Compute layer attributions
#     attributions_lig, delta = lig.attribute(
#         input_ids,
#         target=0,
#         n_steps=50,
#         return_convergence_delta=True,
#         additional_forward_args=attention_mask
#     )
#
#     # Remove interpretable embedding when done
#     remove_interpretable_embedding_layer(model, interpretable_embedding)

def visualize_captum_graphs(request):
    anomaly_finder_id = request.session.get("anomaly_finder_id")

    captum_attr_data = CaptumAttentionData.objects.filter(anomaly_finder_id=anomaly_finder_id).first()
    logs = UploadLog.objects.filter(anomaly_finder_id=anomaly_finder_id).first()
    pred_class = logs[0]['predicted_class'][0]

    attr_sum = captum_attr_data['attributions'][0]
    delta = captum_attr_data['delta'][0]

    inputs = anomaly_detect_model_class.tokenizer(
        logs[0],
        return_tensors="pt",
        truncation=True,
        padding=True
    )
    tokens = anomaly_detect_model_class.tokenizer.convert_ids_to_tokens(inputs[0])

    vis_data_records = []
    vis_data_records.append(viz.VisualizationDataRecord(
        attr_sum,
        pred_class,
        "Anomaly" if pred_class == 1 else "Normal",
        pred_class,
        "Anomaly" if pred_class == 1 else "Normal",
        tokens,
        delta,
        attr_class=pred_class
    ))

    return viz.visualize_text(vis_data_records, return_html=True)

