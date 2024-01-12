from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel, BaseSettings  # Import BaseSettings
from pydantic import BaseModel
from pydantic_settings import BaseSettings  # Adjusted import for BaseSettings

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

app = FastAPI()

# Add BaseSettings to define configuration for Pydantic
class Settings(BaseSettings):
    model_url: str

    class Config:
        protected_namespaces = ()

class Item(BaseModel):
    model_url: str
    api_key: str
    input_text: str

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

@app.post("/predict")
async def predict(item: Item):
    try:
        huggingface_api = HuggingFaceAPI(item.model_url, item.api_key)

        # Example usage with additional tokenizer parameters
        result = huggingface_api.query(item.input_text, max_length=25, padding="max_length", truncation=True)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
