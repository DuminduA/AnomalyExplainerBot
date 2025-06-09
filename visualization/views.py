from django.shortcuts import render

from anomaly_detecter_model.anomaly_detection_roberta_model import AnomalyDetectionRobertaModel
# from explainer.captum_service import save_feature_attribution_with_ig, visualize_captum_graphs
from explainer.captum_service import visualize_log_attribution
from explainer.bertviz_service import get_bertviz_visualizations, get_model_visualization, process_model_attentions, \
    filter_model_attentions
from uploader.models import UploadLog
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

        filtered_model_attentions = filter_model_attentions(model_attentions, anomaly_finder_id)
        html_str_collection, model_view_str_collection, logs = process_model_attentions(filtered_model_attentions,
                                                                                         anomaly_finder_id,
                                                                                         [],
                                                                                         [],
                                                                                         [])

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
        anomaly_finder_id = request.session.get("anomaly_finder_id")
        # save_feature_attribution_with_ig(request)

        html = visualize_log_attribution(anomaly_finder_id)

        return render(request, "visualizations/captum.html", {"html": html})
    return render(request, "visualizations/captum.html", {"html": "<h1>Could not generate the graphs</h1>"})



