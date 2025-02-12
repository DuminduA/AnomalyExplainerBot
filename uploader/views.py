import os

from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForSequenceClassification

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from django.shortcuts import render
from django.conf import settings

torch.device("cpu")

def upload(request):
    login(settings.HUGGING_FACE_WRITE_API_KEY)

    model = AutoModelForSequenceClassification.from_pretrained("Dumi2025/log-anomaly-detection-model")
    tokenizer = AutoTokenizer.from_pretrained("Dumi2025/log-anomaly-detection-model")

    model.to("cpu")
    model.eval()

    # Sample input text
    input_text = "10.251.30.85:50010 Starting thread to transfer block blk_-7057732666118938934 to 10.251.106.214:50010"

    # Tokenize the input
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)

    # Perform classification
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # Get raw scores
        predicted_class = torch.argmax(logits, dim=-1).item()  # Get class index

    # Print results
    print(f"Predicted Class: {predicted_class}")


    return render(request, 'uploader/uploader.html')
