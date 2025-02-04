from django.shortcuts import render
import os

def upload(request):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Specify the model directory (where you saved your trained model)
    model_path = "/Users/duminduakalanka/Documents/Oulu/Thesis/saved_model"
    model_path = os.path.expanduser("~/saved_model")
    tokenizer_path = os.path.expanduser("~/saved_tokenizer")

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("/Users/duminduakalanka/Documents/Oulu/Thesis/saved_tokenizer")

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu", load_in_8bit=False)

    # Sample input text
    input_text = "10.251.30.85:50010 Starting thread to transfer block blk_-7057732666118938934 to 10.251.106.214:50010"

    # Tokenize the input
    inputs = tokenizer(input_text, return_tensors="pt")

    # Generate output
    output = model.generate(**inputs, max_length=50)

    # Decode the output
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    print("Predicted Output:", generated_text)

    return render(request, 'uploader/uploader.html')
