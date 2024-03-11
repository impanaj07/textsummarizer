from flask import Flask, render_template, request,send_file
import requests
import os
import io
from base64 import b64encode
from PIL import Image

API_KEY = "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"
TEXT_URL = "https://api-inference.huggingface.co/models/google/pegasus-large"
IMAGE_URL = "https://api-inference.huggingface.co/models/nerijs/pixel-art-xl"
app = Flask(__name__)
directory_path = 'uploads'
if not os.path.exists(directory_path):
    os.makedirs(directory_path)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route("/")
def home():
    return render_template('index.html')

@app.route("/image_generation")
def imager():
    return render_template("image_generation.html")




@app.route("/text-summarization", methods=["POST"])
def summarize():

    if request.method == "POST":

        inputtext = request.form["inputtext_"]
        payload = {"inputs": inputtext}
        response = requests.post(TEXT_URL, headers=headers, json=payload)
        summary_json = response.json()
        summary = summary_json[0]["summary_text"]
        return render_template("output.html", data={"summary": summary})

@app.route("/text-to-pixel-art", methods=["POST"])
def query():
    if request.method == "POST":
        headers = {"Authorization": f"Bearer {API_KEY}"}
        text = request.form["imageinput"]
        payload = {"inputs": f"pixel art,{text}"}
        response = requests.post(IMAGE_URL, headers=headers, json=payload)
        image_bytes = response.content
        image = Image.open(io.BytesIO(image_bytes))
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        dataurl = "data:image/png;base64," + b64encode(buffered.getvalue()).decode(
            "ascii"
        )
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'generated_image.png')
        image.save(image_path)

        return render_template("output.html", data={"image": dataurl})
@app.route("/download_image")
def download_image():
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'generated_image.png')
    return send_file(image_path, as_attachment=True)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port="4000")


# incase of error use post as 0.0.0.0
