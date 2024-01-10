import sys
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class HuggingFaceAPI:
    def __init__(self, model_url, api_key):
        self.API_URL = model_url
        self.headers = {"Authorization": f"Bearer {api_key}"}
        # Now loading the tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_url)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_url)

    def query(self, input_text):
        # Tokenize the input text
        inputs = self.tokenizer(input_text, return_tensors="pt")
       
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits
       
        return logits.tolist()

def main():
    if len(sys.argv) != 4:
        print("Usage: python tokenizer.py <model_url> <api_key> <input_text>")
        sys.exit(1)

    model_url = sys.argv[1]
    api_key = sys.argv[2]
    input_text = sys.argv[3]

    huggingface_api = HuggingFaceAPI(model_url, api_key)

    logits = huggingface_api.query(input_text)
    print("Logits:", logits)

if __name__ == "__main__":
    main()
