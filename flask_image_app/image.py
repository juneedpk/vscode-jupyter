from flask import Flask, render_template, request
import requests
import base64

app = Flask(__name__)

API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
HEADERS = {"Authorization": "Bearer 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"}

def query(payload):
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    return response.content

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.form['prompt']
    negative_prompt = request.form.get('negative_prompt')
    num_steps = int(request.form.get('num_steps', 50))  # Default to 50 steps if not provided

    # Construct the prompt based on positive and negative prompts
    if negative_prompt:
        prompt = f"not {negative_prompt} and {prompt}"
    else:
        prompt = f"not ugly and deformed and {prompt}"  # Default negative prompt if not provided

    # Query the Hugging Face API with the specified number of steps
    response_content = query({
        "inputs": prompt,
        "num_return_sequences": 3,
        "num_samples": 3,
        "num_steps": num_steps
    })

    # Convert response content to base64
    image_base64 = base64.b64encode(response_content).decode('utf-8')

    return render_template('result.html', image_base64=image_base64)

if __name__ == '__main__':
    app.run(debug=True)
    



