from flask import Flask, render_template
from controller import *

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "Hello, World!"

@app.route("/explain")
def explain():
    return explain("I am a very shy person and I prefer to be alone.", "IE")

@app.route("/predict")
def predict():
    return predict("I am a very shy person and I prefer to be alone.")

@app.route('/plot')
def display_shap_plot():
    plot("I am a very shy person and I prefer to be alone.", "IE")
    return render_template('plot.html')