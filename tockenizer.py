import sys
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class HuggingFaceAPI:
    def __init__(self, model_url, api_key):
        self.API_URL = model_url
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.tokenizer = AutoTokenizer.from_pretrained(model_url)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_url)

    def tokenize_text(self, text, max_length=128, padding=True, truncation=True, return_tensors="pt"):
        inputs = self.tokenizer(
            text,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors
        )
        return inputs

    def query(self, input_text, **tokenizer_args):
        inputs = self.tokenize_text(input_text, **tokenizer_args)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits
        
        return logits.tolist()

def main():
    if len(sys.argv) != 4:
        print("Usage: python script.py <model_url> <api_key> <input_text>")
        sys.exit(1)

    model_url = sys.argv[1]
    api_key = sys.argv[2]
    input_text = sys.argv[3]

    huggingface_api = HuggingFaceAPI(model_url, api_key)

    # Example usage with additional tokenizer parameters
    logits = huggingface_api.query(input_text, max_length=256, padding="max_length", truncation=True)
    print("Logits:", logits)

if __name__ == "__main__":
    main()
