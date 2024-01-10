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

Usage
Clone the repository:
git clone <repository_url>
cd <repository_directory>

Run the script with the following command:
python tockenizer.py <model_url> <api_key> <input_text>
