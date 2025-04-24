import torch.nn.functional as F
import torch
from transformers import RobertaTokenizerFast, BertForSequenceClassification
from captum.attr import visualization as viz, remove_interpretable_embedding_layer, \
    configure_interpretable_embedding_layer
from captum.attr import LayerIntegratedGradients
from django.utils.safestring import mark_safe
from uploader.models import UploadLog
from visualization.models import AnomalyFinderId

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_path = 'Dumi2025/log-anomaly-detection-model-new'

# load model
model = BertForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.eval()
model.zero_grad()

# load tokenizer
tokenizer = RobertaTokenizerFast.from_pretrained(model_path)


# Fix in captum_service.py line 133
def predict(input_ids, attention_mask=None):
    # Add batch dimension if missing
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)  # From [seq_len] to [1, seq_len]
        attention_mask = attention_mask.unsqueeze(0)

    return model(input_ids=input_ids, attention_mask=attention_mask).logits

def forward_func(input_ids, attention_mask=None):
    logits = predict(input_ids, attention_mask)
    # Assuming class 1 is "anomaly"
    return F.softmax(logits, dim=-1)[:, 1]

def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    return attributions / torch.norm(attributions)

def visualize_log_attribution_old(request):
    log_text = UploadLog.objects.first().logs[0]
    # Tokenize
    inputs = tokenizer(log_text, return_tensors="pt", truncation=True, padding=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # Hook embedding layer
    # interpretable_emb = configure_interpretable_embedding_layer(model, 'bert.embeddings')
    lig = LayerIntegratedGradients(forward_func, model.bert.embeddings)

    # Compute attributions
    attributions, delta = lig.attribute(
        inputs=input_ids,
        additional_forward_args=(attention_mask,),
        return_convergence_delta=True,
        n_steps=50
    )

    # remove_interpretable_embedding_layer(model, interpretable_emb)

    # Normalize
    attr_sum = summarize_attributions(attributions)

    # Get prediction
    with torch.no_grad():
        pred_class = torch.argmax(predict(input_ids, attention_mask)).item()
        pred_prob = F.softmax(predict(input_ids, attention_mask), dim=1)[0][pred_class].item()

    # Convert delta to a float if it's a tensor
    if isinstance(delta, torch.Tensor):
        delta = delta.item()

        # Visualization
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


def visualize_log_attribution(anomaly_finder_id):

    logs = UploadLog.objects.filter(anomaly_finder_id=anomaly_finder_id).first().logs
    visualizations = []

    for log_text in logs:
        inputs = tokenizer(log_text, return_tensors="pt", truncation=True, padding=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

        # Attribution computation
        lig = LayerIntegratedGradients(forward_func, model.bert.embeddings)
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

        # Convert delta
        delta_val = delta.item() if isinstance(delta, torch.Tensor) else delta
        label = "Anomaly" if pred_class == 1 else "Normal"

        # Captum record
        record = viz.VisualizationDataRecord(
            word_attributions=attr_sum,
            pred_prob=pred_prob,
            pred_class=label,
            true_class="",
            attr_class=pred_class,
            attr_score=delta_val,
            raw_input_ids=tokens,
            convergence_score=delta_val
        )

        visualizations.append(record)

    # Combine all visualizations
    html_output = viz.visualize_text(visualizations)
    return mark_safe(html_output._repr_html_())