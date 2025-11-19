import torch
from model_loader import load_model

tokenizer, model = load_model()

def summarize(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=180,
            temperature=0.3,
            top_p=0.9
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)