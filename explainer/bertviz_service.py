import json
import re

import numpy as np
from bertviz import head_view, model_view

from uploader.models import UploadLog
from visualization.models import BertvizAttentionData
from anomaly_detecter_model.anomaly_detection_roberta_model import AnomalyDetectionRobertaModel

anomaly_detect_model_class = AnomalyDetectionRobertaModel()

def process_model_attentions(model_attentions, anomaly_finder_id, html_str_collection, model_view_str_collection, logs, save=False):
    for a in model_attentions:
        inputs = a.input_ids
        attentions = a.get_attention_tensors()
        tokens_init = anomaly_detect_model_class.tokenizer.convert_ids_to_tokens(inputs)

        html, tokens = get_bertviz_visualizations(attentions, tokens_init, anomaly_finder_id, save)

        html_str = html._repr_html_()

        model_html = get_model_visualization(attentions, tokens_init)

        model_html_str = model_html._repr_html_()

        html_str_collection.append(html_str)
        model_view_str_collection.append(model_html_str)
        logs.append(tokens)

    return html_str_collection, model_view_str_collection, logs

def get_bertviz_visualizations(attentions, tokens, anomaly_finder_id, save):
    # TODO Shall we add skip special tokens
    # tokens = anomaly_detect_model_class.tokenizer.convert_ids_to_tokens(inputs.get('input_ids')[0], skip_special_tokens=True)

    html = head_view(attentions, tokens, html_action="return")
    if save:
        save_bertviz_head_view(html.data, anomaly_finder_id)
    return html, ' '.join(tokens)

def get_model_visualization(attentions, tokens):
    html = model_view(attentions, tokens, html_action="return")
    return html

def save_bertviz_head_view(head_view_html, anomaly_finder_id):
    # This is looking for a params variable exactly, fragile way to pull the data, but continuing for now
    match = re.search(r'const\s+params\s*=\s*({.*?})\s*;', head_view_html, re.DOTALL)

    if match:
        params_str = match.group(1)

        try:
            params = json.loads(params_str)

            attentions = params["attention"][0]
            attentions.pop("name")
            attentions.pop("right_text")
            attentions["tokens"] = attentions.pop("left_text")

            attention_data = BertvizAttentionData(tokens=attentions["tokens"], attn=attentions['attn'], anomaly_finder_id=anomaly_finder_id)
            attention_data.save()

            return attentions
        except json.JSONDecodeError as e:
            print("Couldn't parse params JSON:", e)
    else:
        print("Couldn't find 'params' in the HTML.")
        return None

def summarize_attention_data_old(attentions, tokens, top_k=5, special_tokens=["[CLS]", "[SEP]", "<s>"]):
    """
    Generate a human-readable summary of attention insights from BERT.

    Parameters:
        attentions (list of np.array): List of [num_heads, seq_len, seq_len] arrays per layer
        tokens (list of str): Tokenized input
        top_k (int): Number of top items to return
        special_tokens (list of str): Tokens to check for bias

    Returns:
        str: Summary of attention insights
    """
    num_layers = len(attentions)
    num_heads = len(attentions[0])
    seq_len = len(tokens)

    def get_most_attended_tokens():
        token_attention_scores = np.zeros(seq_len)
        for layer_attention in attentions:
            for head_attention in layer_attention:
                token_attention_scores += np.array(head_attention).mean(axis=0)
        token_attention_scores /= (num_layers * num_heads)
        top_indices = np.argsort(token_attention_scores)[-top_k:][::-1]
        return [(tokens[i], round(token_attention_scores[i], 3)) for i in top_indices]

    def get_most_focused_heads():
        focused_heads = []
        for layer_idx, layer_attention in enumerate(attentions):
            for head_idx, head_attention in enumerate(layer_attention):
                head_array = np.array(head_attention)
                entropies = -np.sum(head_array * np.log(head_array + 1e-9), axis=-1)
                avg_entropy = np.mean(entropies)
                focused_heads.append((layer_idx, head_idx, round(avg_entropy, 3)))
        return sorted(focused_heads, key=lambda x: x[2])[:top_k]

    def get_standout_layers():
        layer_scores = []
        for layer_idx, layer_attention in enumerate(attentions):
            total_focus = 0
            for head_attention in layer_attention:
                head_array = np.array(head_attention)
                entropies = -np.sum(head_attention * np.log(head_array + 1e-9), axis=-1)
                total_focus += (1 / (np.mean(entropies) + 1e-9))
            avg_focus = total_focus / num_heads
            layer_scores.append((layer_idx, round(avg_focus, 3)))
        return sorted(layer_scores, key=lambda x: x[1], reverse=True)[:top_k]

    def detect_special_token_bias(threshold=0.3):
        bias_info = []
        for layer_idx, layer_attention in enumerate(attentions):
            for head_idx, head_attention in enumerate(layer_attention):
                head_array = np.array(head_attention)
                for special_token in special_tokens:
                    if special_token in tokens:
                        idx = tokens.index(special_token)
                        avg_focus = head_array[:, idx].mean()
                        if avg_focus > threshold:
                            bias_info.append(
                                f"Layer {layer_idx} Head {head_idx} over-focuses on '{special_token}' (avg: {avg_focus:.2f})"
                            )
        return bias_info

    # Build report
    report = []

    report.append("📌 Top Attended Tokens:")
    for tok, score in get_most_attended_tokens():
        report.append(f"- \"{tok}\": {score}")

    report.append("\n🔍 Most Focused Heads (Low Entropy):")
    for layer, head, entropy in get_most_focused_heads():
        report.append(f"- Layer {layer} Head {head} (entropy: {entropy})")

    report.append("\n⭐ Standout Layers (High Focus):")
    for layer, score in get_standout_layers():
        report.append(f"- Layer {layer} (focus score: {score})")

    bias_info = detect_special_token_bias()
    if bias_info:
        report.append("\n🚨 Special Token Bias:")
        report.extend(f"- {bias}" for bias in bias_info)

    return "\n".join(report)


def summarize_attention_data(attentions, tokens, top_k=5,
                             special_tokens=["[CLS]", "[SEP]", "<s>", "</s>", "<pad>", "[PAD]", "<unk>", "[UNK]"]):
    """
    Generate a human-readable summary of attention insights from BERT, avoiding special tokens.

    Parameters:
        attentions (list of np.array): List of [num_heads, seq_len, seq_len] arrays per layer
        tokens (list of str): Tokenized input
        top_k (int): Number of top items to return
        special_tokens (list of str): Tokens to exclude from analysis

    Returns:
        str: Summary of attention insights
    """
    num_layers = len(attentions)
    num_heads = len(attentions[0])
    seq_len = len(tokens)

    # Create mask for non-special tokens
    non_special_mask = [i for i, token in enumerate(tokens) if token not in special_tokens]

    def get_most_attended_tokens():
        token_attention_scores = np.zeros(seq_len)
        for layer_attention in attentions:
            for head_attention in layer_attention:
                token_attention_scores += np.array(head_attention).mean(axis=0)
        token_attention_scores /= (num_layers * num_heads)

        # Filter out special tokens
        filtered_scores = [(i, token_attention_scores[i]) for i in non_special_mask]
        top_indices = sorted(filtered_scores, key=lambda x: x[1], reverse=True)[:top_k]

        return [(tokens[i], round(score, 3)) for i, score in top_indices]

    def get_most_focused_heads():
        focused_heads = []
        for layer_idx, layer_attention in enumerate(attentions):
            for head_idx, head_attention in enumerate(layer_attention):
                head_array = np.array(head_attention)

                # Calculate entropy only for non-special token positions
                filtered_attention = head_array[:, non_special_mask]
                if filtered_attention.shape[1] > 0:  # Ensure we have non-special tokens
                    entropies = -np.sum(filtered_attention * np.log(filtered_attention + 1e-9), axis=-1)
                    avg_entropy = np.mean(entropies)
                    focused_heads.append((layer_idx, head_idx, round(avg_entropy, 3)))

        return sorted(focused_heads, key=lambda x: x[2])[:top_k]

    def get_standout_layers():
        layer_scores = []
        for layer_idx, layer_attention in enumerate(attentions):
            total_focus = 0
            valid_heads = 0

            for head_attention in layer_attention:
                head_array = np.array(head_attention)

                # Calculate focus only for non-special token positions
                filtered_attention = head_array[:, non_special_mask]
                if filtered_attention.shape[1] > 0:  # Ensure we have non-special tokens
                    entropies = -np.sum(filtered_attention * np.log(filtered_attention + 1e-9), axis=-1)
                    total_focus += (1 / (np.mean(entropies) + 1e-9))
                    valid_heads += 1

            if valid_heads > 0:
                avg_focus = total_focus / valid_heads
                layer_scores.append((layer_idx, round(avg_focus, 3)))

        return sorted(layer_scores, key=lambda x: x[1], reverse=True)[:top_k]

    def detect_special_token_bias(threshold=0.3):
        bias_info = []
        for layer_idx, layer_attention in enumerate(attentions):
            for head_idx, head_attention in enumerate(layer_attention):
                head_array = np.array(head_attention)
                for special_token in special_tokens:
                    if special_token in tokens:
                        idx = tokens.index(special_token)
                        avg_focus = head_array[:, idx].mean()
                        if avg_focus > threshold:
                            bias_info.append(
                                f"Layer {layer_idx} Head {head_idx} over-focuses on '{special_token}' (avg: {avg_focus:.2f})"
                            )
        return bias_info

    # Build report
    report = []

    report.append("📌 Top Attended Tokens:")
    for tok, score in get_most_attended_tokens():
        report.append(f"- \"{tok}\": {score}")

    report.append("\n🔍 Most Focused Heads (Low Entropy):")
    for layer, head, entropy in get_most_focused_heads():
        report.append(f"- Layer {layer} Head {head} (entropy: {entropy})")

    report.append("\n⭐ Standout Layers (High Focus):")
    for layer, score in get_standout_layers():
        report.append(f"- Layer {layer} (focus score: {score})")

    bias_info = detect_special_token_bias()
    if bias_info:
        report.append("\n🚨 Special Token Bias:")
        report.extend(f"- {bias}" for bias in bias_info)

    return "\n".join(report)

def filter_model_attentions(model_attentions, anomaly_finder_id):
    log_record = UploadLog.objects.filter(anomaly_finder_id=anomaly_finder_id).get()
    filtered_list = []
    for lidx, log in enumerate(log_record.logs):
        for idx, attn in enumerate(model_attentions):
            if attn.log == log and log_record.predicted_class[idx]==1:
                filtered_list.append(attn)
            elif lidx == len(log_record.logs) - 1 and idx == len(model_attentions) -2:
                filtered_list.append(attn)
    return filtered_list