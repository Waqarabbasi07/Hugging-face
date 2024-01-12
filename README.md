# Hugging Face Inference Script

This script demonstrates how to use the Hugging Face Inference API for sequence classification models. It takes user input, tokenizes it, and queries a Hugging Face model for inference.

## Prerequisites

Before running the script, make sure you have the following installed:

- Python (>=3.6)
- pip (package installer for Python)

Install required Python packages using:

```bash
pip install requests transformers torch
Using Transformers>=4.13 might solve some issues.
```

## Usage
Clone the repository:

```
git clone <repository_url>
cd <repository_directory>
```

** Run the script with the following command: **
```python tockenizer.py <model_url> <api_key> <input_text>```

## Output
The script outputs the raw logits for each class. These logits represent the model's confidence scores for each class. You can interpret these scores based on your specific use case.

## json response of fast api
```
{
  "inputs": {
    "input_ids": [
      1,
      584,
      464,
      278,
      304,
      343,
      281,
      991,
      479,
      264,
      2965,
      310,
      2,
      0,
      0,
      0
   
    ],
    "token_type_ids": [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    "attention_mask": [
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0
    ]
  },
  "logits": [
    -1.6624224185943604,
    3.6730949878692627,
    -2.5786781311035156
  ],
  "probabilities": [
    0.0047851442359387875,
    0.9933007955551147,
    0.001914125052280724
  ],
  "labels": {
    "0": "entailment",
    "1": "neutral",
    "2": "contradiction"
  }
}
```
