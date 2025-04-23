from django.shortcuts import render

from anomaly_detecter_model.anomaly_detection_roberta_model import AnomalyDetectionRobertaModel
from explainer.captum_service import save_feature_attribution_with_ig, visualize_captum_graphs
from explainer.bertviz_service import get_bertviz_visualizations, get_model_visualization
from visualization.models import ModelAttentions

anomaly_detect_model_class = AnomalyDetectionRobertaModel()

def bert_attention_view(request):
    if request.method == "GET":
        print("Bertviz visualization...")
        anomaly_finder_id = request.session.get("anomaly_finder_id")

        if not anomaly_finder_id:
            return render(request, "visualizations/bertviz.html",
                          {"graphs": "<h1>Could not generate the graphs, No log file uploaded for this session. </h1>",
                           'model_view': "", 'logs': ""})


        model_attentions = ModelAttentions.objects.filter(anomaly_finder_id=anomaly_finder_id)
        if not model_attentions:
            return render(request, "visualizations/bertviz.html",
                          {"graphs": "<h1>Could not generate the graphs, No bertviz model attentions</h1>", 'model_view': "", 'logs': ""})

        html_str_collection = []
        model_view_str_collection = []
        logs = []

        for a in model_attentions:
            inputs = a.input_ids
            attentions = a.get_attention_tensors()

            html, tokens = get_bertviz_visualizations(attentions, inputs, anomaly_finder_id)

            html_str = html._repr_html_()

            model_html = get_model_visualization(attentions, inputs)

            model_html_str = model_html._repr_html_()

            html_str_collection.append(html_str)
            model_view_str_collection.append(model_html_str)
            logs.append(tokens)

        paired_data = [
            {"graph": graph, "mv": mv}
            for graph, mv in zip(html_str_collection, model_view_str_collection)
        ]

        context = {
            "logs": logs,
            "paired_data": paired_data
        }

        return render(request, "visualizations/bertviz.html", context)
    return render(request, "visualizations/bertviz.html", {"graphs": "<h1>Could not generate the graphs</h1>", 'model_view': "", 'logs': ""})



from captum.attr import IntegratedGradients


ig = IntegratedGradients(anomaly_detect_model_class.model)
def captum_attention_view(request):
    if request.method == "GET":
        print("Captum visualization...")
        save_feature_attribution_with_ig(request)

        html = visualize_captum_graphs(request)

        return render(request, "visualizations/captum.html", {"html": html})
    return render(request, "visualizations/captum.html", {"html": "<h1>Could not generate the graphs</h1>"})



