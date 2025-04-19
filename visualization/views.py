from django.http import JsonResponse
from django.shortcuts import render
from bertviz import head_view, model_view
from uploader.views import UploaderViewSet
import re, json
import matplotlib.pyplot as plt
import numpy as np

from visualization.models import AttentionData

anomaly_detect_model_class = UploaderViewSet.anomaly_detect_model_class

def bert_attention_view(request):
    if request.method == "GET":
        print("Bertviz visualization...")
        if len(anomaly_detect_model_class.attentions):

            html_str_collection = []
            model_view_str_collection = []
            logs = []
            anomaly_finder_id = request.session.get("anomaly_finder_id")

            if not anomaly_finder_id:
                return render(request, "visualizations/bertviz.html",
                              {"graphs": "<h1>Could not generate the graphs, No log file uploaded for this session. </h1>", 'model_view': "", 'logs': ""})

            for a in anomaly_detect_model_class.attentions:
                inputs = a['inputs']
                attentions = a['attentions']

                html, tokens = get_bertviz_visualizations(attentions, inputs, anomaly_finder_id)

                html_str = html._repr_html_()

                model_html = get_model_visualization(attentions, inputs)

                model_html_str = model_html._repr_html_()

                html_str_collection.append(html_str)
                model_view_str_collection.append(model_html_str)
                logs.append(tokens)

            return render(request, "visualizations/bertviz.html", {"graphs": html_str_collection, 'model_view': model_view_str_collection, 'logs': logs})
        return render(request, "visualizations/bertviz.html", {"graphs": "<h1>Could not generate the graphs</h1>", 'model_view': "", 'logs': ""})

def get_bertviz_visualizations(attentions, inputs, anomaly_finder_id):
    tokens = anomaly_detect_model_class.tokenizer.convert_ids_to_tokens(inputs.get('input_ids')[0], skip_special_tokens=True)

    html = head_view(attentions, tokens, html_action="return")
    save_bertviz_head_view(html.data, anomaly_finder_id)
    return html, ' '.join(tokens)

def get_model_visualization(attentions, inputs):
    tokens = anomaly_detect_model_class.tokenizer.convert_ids_to_tokens(inputs.get('input_ids')[0], skip_special_tokens=True)

    html = model_view(attentions, tokens, html_action="return")
    return html

def save_bertviz_head_view(head_view_html, anomaly_finder_id):
    # This is looking for a params variable exactly, fragile way to pull the data, but continuing for now
    match = re.search(r'const\s+params\s*=\s*({.*?})\s*;', head_view_html, re.DOTALL)

    if match:
        params_str = match.group(1)

        try:
            params = json.loads(params_str)
            print("Successfully parsed params!")

            attentions = params["attention"][0]
            attentions.pop("name")
            attentions.pop("right_text")
            attentions["tokens"] = attentions.pop("left_text")

            summarized_attentions = summarize_attention_data(attentions["attn"], len(attentions["tokens"]))

            attention_data = AttentionData(tokens=attentions["tokens"], attn=summarized_attentions, anomaly_finder_id=anomaly_finder_id)
            attention_data.save()

            return attentions
        except json.JSONDecodeError as e:
            print("Couldn't parse params JSON:", e)
    else:
        print("Couldn't find 'params' in the HTML.")
        return None

def summarize_attention_data(attentions, num_tokens):
    num_layers = len(attentions)
    num_heads = len(attentions[0])

    top_k = 5
    top_indices_structure = []

    for layer in range(num_layers):
        layer_list = []
        for head in range(num_heads):
            head_list = []
            for token_idx in range(num_tokens):
                attn_vector = attentions[layer][head][token_idx]
                top_indices = list(np.argsort(attn_vector)[-top_k:][::-1])
                head_list.append(top_indices)
            layer_list.append(head_list)
        top_indices_structure.append(layer_list)

    return top_indices_structure



from captum.attr import IntegratedGradients

ig = IntegratedGradients(anomaly_detect_model_class.model)
def captum_attention_view(request):
    if request.method == "GET":
        print("Captum visualization...")
        if len(anomaly_detect_model_class.attentions):

            html_str_collection = []
            model_view_str_collection = []
            logs = []

            for a in anomaly_detect_model_class.attentions:
                inputs = a['inputs']
                attentions = a['attentions']
                input_ids = inputs['input_ids'].long()
                attributions = ig.attribute(input_ids, target=1)

                tokens = anomaly_detect_model_class.tokenizer.convert_ids_to_tokens(inputs[0])

                # Normalize attributions
                attributions = attributions.squeeze().detach().numpy()
                attributions = (attributions - attributions.min()) / (
                            attributions.max() - attributions.min())  # Normalize to [0,1]

                plt.figure(figsize=(12, 4))
                plt.barh(tokens, attributions, color="skyblue")
                plt.xlabel("Attribution Score")
                plt.ylabel("Token")
                plt.title("Token Attributions for Log Anomaly Detection")
                plt.gca().invert_yaxis()
                plt.show()

            return render(request, "visualizations/captum.html", {"graphs": html_str_collection, 'model_view': model_view_str_collection, 'logs': logs})
        return render(request, "visualizations/captum.html", {"graphs": "<h1>Could not generate the graphs</h1>", 'model_view': "", 'logs': ""})

def preprocess_log(log_text):
    inputs = anomaly_detect_model_class.tokenizer(log_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    return inputs.input_ids, inputs.attention_mask
def interpret_log(request):
    log_text = request.GET.get("log", "ERROR: Connection timeout after multiple retries.")

    # Tokenize log
    input_ids, attention_mask = preprocess_log(log_text)

    # Compute attributions
    ig = IntegratedGradients(anomaly_detect_model_class)
    attributions = ig.attribute(input_ids, target=1).squeeze().detach().numpy()

    # Normalize attributions
    attributions = (attributions - attributions.min()) / (attributions.max() - attributions.min())

    # Convert tokens and attributions to JSON format
    tokens = anomaly_detect_model_class.tokenizer.convert_ids_to_tokens(input_ids[0])
    data = {"tokens": tokens, "attributions": attributions.tolist()}

    return JsonResponse(data)




