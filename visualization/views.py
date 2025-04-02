from django.shortcuts import render
from bertviz import head_view, model_view
from uploader.views import UploaderViewSet

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

def get_bertviz_visualizations(attentions, inputs):
    tokens = model.tokenizer.convert_ids_to_tokens(inputs.get('input_ids')[0])

    html = head_view(attentions, tokens, html_action="return")
    return html, ' '.join(tokens)

def get_model_visualization(attentions, inputs):
    tokens = model.tokenizer.convert_ids_to_tokens(inputs.get('input_ids')[0])

    html = model_view(attentions, tokens, html_action="return")
    return html

