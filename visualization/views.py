from django.http import JsonResponse
from django.shortcuts import render
from bertviz import head_view, model_view
from uploader.views import UploaderViewSet
import re, json
import matplotlib.pyplot as plt

model = UploaderViewSet.model

def bert_attention_view(request):
    if request.method == "GET":
        print("Bertviz visualization...")
        if len(model.attentions):

            html_str_collection = []
            model_view_str_collection = []
            logs = []

            for a in model.attentions:
                inputs = a['inputs']
                attentions = a['attentions']

                html, tokens = get_bertviz_visualizations(attentions, inputs)

                html_str = html._repr_html_()

                model_html = get_model_visualization(attentions, inputs)

                model_html_str = model_html._repr_html_()

                html_str_collection.append(html_str)
                model_view_str_collection.append(model_html_str)
                logs.append(tokens)

            return render(request, "visualizations/bertviz.html", {"graphs": html_str_collection, 'model_view': model_view_str_collection, 'logs': logs})
        return render(request, "visualizations/bertviz.html", {"graphs": "<h1>Could not generate the graphs</h1>", 'model_view': "", 'logs': ""})


from captum.attr import IntegratedGradients

ig = IntegratedGradients(model.model)
def captum_attention_view(request):
    if request.method == "GET":
        print("Captum visualization...")
        if len(model.attentions):

            html_str_collection = []
            model_view_str_collection = []
            logs = []

            for a in model.attentions:
                inputs = a['inputs']
                attentions = a['attentions']

                attributions = ig.attribute(inputs, target=1)

                tokens = model.tokenizer.convert_ids_to_tokens(inputs[0])

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
    inputs = model.tokenizer(log_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    return inputs.input_ids, inputs.attention_mask
def interpret_log(request):
    log_text = request.GET.get("log", "ERROR: Connection timeout after multiple retries.")

    # Tokenize log
    input_ids, attention_mask = preprocess_log(log_text)

    # Compute attributions
    ig = IntegratedGradients(model)
    attributions = ig.attribute(input_ids, target=1).squeeze().detach().numpy()

    # Normalize attributions
    attributions = (attributions - attributions.min()) / (attributions.max() - attributions.min())

    # Convert tokens and attributions to JSON format
    tokens = model.tokenizer.convert_ids_to_tokens(input_ids[0])
    data = {"tokens": tokens, "attributions": attributions.tolist()}

    return JsonResponse(data)

def get_bertviz_visualizations(attentions, inputs):
    tokens = model.tokenizer.convert_ids_to_tokens(inputs.get('input_ids')[0])

    html = head_view(attentions, tokens, html_action="return")
    convert_bertviz_head_view_to_json_data(html.data)
    return html, ' '.join(tokens)

def get_model_visualization(attentions, inputs):
    tokens = model.tokenizer.convert_ids_to_tokens(inputs.get('input_ids')[0])

    html = model_view(attentions, tokens, html_action="return")
    return html

def convert_bertviz_head_view_to_json_data(head_view_html):
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

            return attentions
        except json.JSONDecodeError as e:
            print("Couldn't parse params JSON:", e)
    else:
        print("Couldn't find 'params' in the HTML.")
        return None




