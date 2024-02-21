from flask import Flask, render_template, request

from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch

app = Flask(__name__)

model_name = "google/pegasus-xsum"
tokenizer = PegasusTokenizer.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/text-summarization', methods=["POST"])
def summarize():

    if request.method == "POST":

        inputtext = request.form["inputtext_"]

        input_text = "summarize: " + inputtext

        tokenized_text = tokenizer.encode(input_text, return_tensors='pt', max_length=512).to(device)
        summary_ = model.generate(tokenized_text, min_length=30, max_length=500)
        summary = tokenizer.decode(summary_[0], skip_special_tokens=True)

        '''
            text = <start> i am yash <end>
            vocab = { i: 1, am : 2, yash: 3, start 4}

            token = [i, am ,yash]
            encode = [1 2, 3, 4]

            summary_ = [[4, 3,1, 5]]

            summary = yash i

        
        '''

    return render_template("output.html", data = {"summary": summary})

if __name__ == '__main__':
    app.run(host="192.168.1.12",port="5000")


# incase of error use post as 0.0.0.0
