from flask import Flask
from controller import *

app = Flask(__name__)

@app.route("/")
def hello_world():
    return explain("I am a very outgoing person.", "IE")

@app.route("/predict")
def predict():
    return predict("I am a very outgoing person.")