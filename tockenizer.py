import sys
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import torch
import json

class HuggingFaceAPI:
    def __init__(self, model_url, api_key):
        self.API_URL = model_url
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.tokenizer = AutoTokenizer.from_pretrained(model_url)
        self.config = AutoConfig.from_pretrained(model_url)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_url, config=self.config)

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

        # Convert BatchEncoding to dictionary
        inputs_dict = {key: value.squeeze().tolist() for key, value in inputs.items()}

        # Softmax to get probabilities
        probabilities = torch.nn.functional.softmax(logits, dim=-1)

        # Get label names
        labels = self.config.id2label

        result = {
            "inputs": inputs_dict,
            "logits": logits.squeeze().tolist(),
            "probabilities": probabilities.squeeze().tolist(),
            "labels": labels,
        }

        return result

def main():
    if len(sys.argv) != 4:
        print("Usage: python script.py <model_url> <api_key> <input_text>")
        sys.exit(1)

    model_url = sys.argv[1]
    api_key = sys.argv[2]
    input_text = sys.argv[3]

    huggingface_api = HuggingFaceAPI(model_url, api_key)

    # Example usage with additional tokenizer parameters
    result = huggingface_api.query(input_text, max_length=25, padding="max_length", truncation=True)
    result_json = json.dumps(result, indent=2)
    print(result_json)

if __name__ == "__main__":
    main()
