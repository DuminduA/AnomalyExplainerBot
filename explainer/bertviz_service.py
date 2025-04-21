import json
import re

import numpy as np
from bertviz import head_view, model_view

from uploader.views import UploaderViewSet
from visualization.models import BertvizAttentionData

anomaly_detect_model_class = UploaderViewSet.anomaly_detect_model_class

def get_bertviz_visualizations(attentions, inputs, anomaly_finder_id):
    # TODO Shall we add skip special tokens
    # tokens = anomaly_detect_model_class.tokenizer.convert_ids_to_tokens(inputs.get('input_ids')[0], skip_special_tokens=True)
    tokens = anomaly_detect_model_class.tokenizer.convert_ids_to_tokens(inputs)

    html = head_view(attentions, tokens, html_action="return")
    save_bertviz_head_view(html.data, anomaly_finder_id)
    return html, ' '.join(tokens)

def get_model_visualization(attentions, inputs):
    tokens = anomaly_detect_model_class.tokenizer.convert_ids_to_tokens(inputs, skip_special_tokens=False)

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

            attentions['attn'] = summarize_attention_data(attentions["attn"], len(attentions["tokens"]))

            attention_data = BertvizAttentionData(tokens=attentions["tokens"], attn=attentions['attn'], anomaly_finder_id=anomaly_finder_id)
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

    top_k = 2
    top_indices_structure = []

    for layer in range(num_layers):
        layer_list = []
        for head in range(num_heads):
            head_list = []
            for token_idx in range(num_tokens):
                attn_vector = attentions[layer][head][token_idx]
                top_indices = [int(i) for i in np.argsort(attn_vector)[-top_k:][::-1]]
                head_list.append(top_indices)
            layer_list.append(head_list)
        top_indices_structure.append(layer_list)

    return top_indices_structure