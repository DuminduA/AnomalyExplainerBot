import torch.nn.functional as F
import torch
from transformers import RobertaTokenizerFast, BertForSequenceClassification
from captum.attr import visualization as viz, remove_interpretable_embedding_layer, \
    configure_interpretable_embedding_layer
from captum.attr import LayerIntegratedGradients
from django.utils.safestring import mark_safe

from anomaly_detecter_model.anomaly_detection_roberta_model import AnomalyDetectionRobertaModel
from uploader.models import UploadLog
from visualization.models import AnomalyFinderId, CaptumAttentionData

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_path = 'Dumi2025/log-anomaly-detection-model-new'

# load model
model_class = AnomalyDetectionRobertaModel()
model = model_class.model
model.to(device)
model.eval()
model.zero_grad()

tokenizer = model_class.tokenizer


def predict(input_ids, attention_mask=None):
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)

    return model(input_ids=input_ids, attention_mask=attention_mask).logits

def forward_func(input_ids, attention_mask=None):
    logits = predict(input_ids, attention_mask)
    return F.softmax(logits, dim=-1)[:, 1]

def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    return attributions / torch.norm(attributions)

def visualize_log_attribution_old(request):
    log_text = UploadLog.objects.first().logs[0]
    inputs = tokenizer(log_text, return_tensors="pt", truncation=True, padding=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    lig = LayerIntegratedGradients(forward_func, model.roberta.embeddings)

    # Compute attributions
    attributions, delta = lig.attribute(
        inputs=input_ids,
        additional_forward_args=(attention_mask,),
        return_convergence_delta=True,
        n_steps=50
    )

    attr_sum = summarize_attributions(attributions)

    with torch.no_grad():
        pred_class = torch.argmax(predict(input_ids, attention_mask)).item()
        pred_prob = F.softmax(predict(input_ids, attention_mask), dim=1)[0][pred_class].item()

    if isinstance(delta, torch.Tensor):
        delta = delta.item()

    text_and_pred = "Anomaly" if pred_class == 1 else "Normal"

    record = viz.VisualizationDataRecord(
        word_attributions=attr_sum,
        pred_prob=pred_prob,
        pred_class=text_and_pred,
        true_class=text_and_pred,
        attr_class=pred_class,
        attr_score=delta,
        raw_input_ids=tokens,
        convergence_score=delta
    )

    print("Prediction:", "Anomaly" if pred_class == 1 else "Normal")
    html = viz.visualize_text([record])
    return html._repr_html_()


def compute_log_attribution(log_text, anomaly_finder_id):
    """Run Layer Integrated Gradients for one log line and store the result."""
    inputs = tokenizer(log_text, return_tensors="pt", truncation=True, padding=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # Attribution computation (the expensive part: 50 integration steps)
    lig = LayerIntegratedGradients(forward_func, model.roberta.embeddings)
    attributions, delta = lig.attribute(
        inputs=input_ids,
        additional_forward_args=(attention_mask,),
        return_convergence_delta=True,
        n_steps=50
    )

    # Normalize
    attr_sum = summarize_attributions(attributions)

    # Prediction
    with torch.no_grad():
        logits = predict(input_ids, attention_mask)
        pred_class = torch.argmax(logits, dim=-1).item()
        pred_prob = torch.softmax(logits, dim=1)[0][pred_class].item()

    delta_val = delta.item() if isinstance(delta, torch.Tensor) else float(delta)

    record = CaptumAttentionData(
        anomaly_finder_id=anomaly_finder_id,
        log=log_text,
        tokens=tokens,
        attributions=attr_sum.detach().cpu().tolist(),
        delta=delta_val,
        pred_class=pred_class,
        pred_prob=pred_prob,
    )
    record.save()
    return record


def _to_visualization_record(attribution):
    label = "Anomaly" if attribution.pred_class == 1 else "Normal"
    return viz.VisualizationDataRecord(
        word_attributions=torch.tensor(attribution.attributions),
        pred_prob=attribution.pred_prob,
        pred_class=label,
        true_class="",
        attr_class=attribution.pred_class,
        attr_score=attribution.delta,
        raw_input_ids=attribution.tokens,
        convergence_score=attribution.delta
    )


def visualize_log_attribution(anomaly_finder_id):
    """
    Build the Captum page for an analysis. Attributions are computed on demand
    the first time the page is opened and cached in CaptumAttentionData, so
    each log line of an analysis is attributed at most once.
    """
    logs = UploadLog.objects.filter(anomaly_finder_id=anomaly_finder_id).first().logs

    cached = {
        entry.log: entry
        for entry in CaptumAttentionData.objects.filter(anomaly_finder_id=anomaly_finder_id)
    }

    visualizations = []
    for log_text in logs:
        attribution = cached.get(log_text)
        if attribution is None:
            attribution = compute_log_attribution(log_text, anomaly_finder_id)
            cached[log_text] = attribution
        visualizations.append(_to_visualization_record(attribution))

    # Combine all visualizations
    html_output = viz.visualize_text(visualizations)
    return mark_safe(html_output._repr_html_())
