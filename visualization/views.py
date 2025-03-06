from django.shortcuts import render
from bertviz import head_view
from uploader.views import UploaderViewSet

model = UploaderViewSet.model

def bert_attention_view(request):
    if request.method == "GET":
        if len(model.attentions):
            inputs = model.attentions[0]['inputs']
            attentions = model.attentions[0]['attentions']
            html = get_bertviz_visualizations(attentions, inputs)

            html_str = html.__html__()
            return render(request, "visualizations/bertviz.html", {"graphs": html_str})
        return render(request, "visualizations/bertviz.html", {"graphs": "<h1>Could not generate the graphs</h1>"})

def get_bertviz_visualizations(attentions, inputs):
    tokens = model.tokenizer.convert_ids_to_tokens(inputs.get('input_ids')[0])

    html = head_view(attentions, tokens, html_action="return")
    return html

