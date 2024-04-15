import os
import shap
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import json

import joblib
# if os.path.exists('ckpts'):
#     CHECKPOINTPATH = 'ckpts/Persnality_MBTI'
# else:
#     CHECKPOINTPATH = '../../ckpts/Persnality_MBTI'

# ----------------- GLOBAL VARIABLES -----------------

# training parameters
MAX_LEN = 512
# setting the model name
PRE_TRAINED_MODEL_NAME = 'roberta-base'


# TOKENIZER
tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

# setting the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# setting the random seed
torch.manual_seed(99)
torch.cuda.manual_seed(99)
torch.cuda.manual_seed_all(99)
np.random.seed(99)

# ----------------- MODEL CLASS -----------------


class ROBERTAClass(torch.nn.Module):
    def __init__(self, PRE_TRAINED_MODEL_NAME, num_classes=4, dropout=0.3):
        super(ROBERTAClass, self).__init__()
        self.bert_model = AutoModel.from_pretrained(
            PRE_TRAINED_MODEL_NAME, return_dict=True, output_attentions=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert_model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        output_dropout = self.dropout(output.pooler_output)
        output = self.linear(output_dropout)
        return output

    def getProbabilities(self, input):
        input = input.tolist()
        input = tokenizer(input, max_length=MAX_LEN, truncation=True,
                          padding='longest', return_token_type_ids=True, return_tensors='pt')
        input = {key: tensor.to(device) for key, tensor in input.items()}
        output = self.forward(**input)
        probabilities = torch.sigmoid(output)
        return probabilities

    def getBinaryProbs(self, input):
        probabilities = self.getProbabilities(input)
        IE = []
        NS = []
        TF = []
        JP = []
        # Iterate over each item in the probabilities tensor
        for prob in probabilities:
            # Apply the transformation to each item and directly append to the list
            IE.append(torch.tensor([1 - prob[0], prob[0]]))
            NS.append(torch.tensor([1 - prob[1], prob[1]]))
            TF.append(torch.tensor([1 - prob[2], prob[2]]))
            JP.append(torch.tensor([1 - prob[3], prob[3]]))

        # Convert the list of tensors to a single tensor
        IE = torch.stack(IE)
        NS = torch.stack(NS)
        TF = torch.stack(TF)
        JP = torch.stack(JP)
        return IE, NS, TF, JP
    # ----------------- MODEL PREDICTION -----------------

    def predict(self, text):
        return self.getProbabilities(np.array([text])).round()[0].detach().numpy().tolist()

    # ----------------- SHAP EXPLANATION -----------------
    def explain(self, input, aspect):
        def get_aspect_explanation(input):
            IE, NS, TF, JP = self.getBinaryProbs(input)

            if aspect == 'IE':
                return IE
            elif aspect == 'NS':
                return NS
            elif aspect == 'TF':
                return TF
            elif aspect == 'JP':
                return JP
            else:
                raise ValueError("Invalid aspect provided.")

        class_names = self.get_class_names(aspect)
        aspect_explainer = shap.Explainer(
            get_aspect_explanation, masker=shap.maskers.Text(), output_names=class_names)
        shap_values = aspect_explainer(input)
        return shap_values
    # ----------------- HELPER FUNCTIONS -----------------

    def get_class_names(self, aspect):
        if aspect == 'IE':
            return ["Introvert", "Extrovert"]
        elif aspect == 'NS':
            return ["Intuition", "Sensing"]
        elif aspect == 'TF':
            return ["Thinking", "Feeling"]
        elif aspect == 'JP':
            return ["Judging", "Perceiving"]
        else:
            raise ValueError("Invalid aspect provided.")


# ----------------- LOAD MODEL -----------------
model = joblib.load('roberta_mbti_model.pkl')

# ----------------- PREDICTION FUNCTION -----------------


def predict(text):
    return model.predict(text)

# ----------------- SHAP EXPLANATION FUNCTION -----------------

def custom_json_serializer(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert NumPy array to list
    elif isinstance(obj, tuple):
        return list(obj)  # Convert tuple to list
    elif isinstance(obj, np.float32):
        return float(obj)  # Convert np.float32 to Python float
    elif isinstance(obj, np.float64):
        return float(obj)  # Convert np.float64 to Python float
    # Add more conversions as needed
    return obj

def explain(input, aspect):
    shap_values = model.explain([input], aspect)
    shap_values_data = {
        'values': shap_values.values,
        'base_values': shap_values.base_values,
        'data': shap_values.data,
    }

    shap_values_json = json.dumps(shap_values_data, default=custom_json_serializer)
    print(shap_values_json)

    return shap_values_json
