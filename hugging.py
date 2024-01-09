import sys
import requests


print('*****')
class HuggingFaceAPI:
    def __init__(self, model_url, api_key):
        self.API_URL = model_url
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def query(self, payload):
        try:
          response = requests.post(self.API_URL, headers=self.headers, json=payload)
          return response.json()
        except requests.exceptions.RequestException as e:
          print(f"Exception occure:{e}")
          return None


def main():
    if len(sys.argv) != 4:
        print("Usage: python script.py <model_url> <api_key> <input_text>")
        sys.exit(1)

    model_url = sys.argv[1]
    api_key = sys.argv[2]
    input_text = sys.argv[3]

    huggingface_api = HuggingFaceAPI(model_url, api_key)

    payload = {"inputs": input_text, "parameters": {"candidate_labels": ["refund", "legal", "faq"]}}

    output = huggingface_api.query(payload)
    if output is not None:
      print(output)
    else:
      print("Please check your i/p and try again")


if __name__ == "__main__":
    main()
