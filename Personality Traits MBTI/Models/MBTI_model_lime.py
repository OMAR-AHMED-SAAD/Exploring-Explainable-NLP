import re
from nltk.corpus import stopwords
from nltk import word_tokenize
import nltk
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from lime.lime_text import LimeTextExplainer

"""
Content:
- Setup
- Model
- LIME Explanation
- Helper Functions
"""


# ----------------- SETUP -----------------
PRE_TRAINED_MODEL_NAME = 'roberta-base'
MAX_LEN = 512
tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------- SEEDING -----------------
torch.manual_seed(99)
torch.cuda.manual_seed(99)
torch.cuda.manual_seed_all(99)
np.random.seed(99)


# ----------------- MODEL -----------------

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
        input = [clean_text(text) for text in input]
        input = tokenizer(input, max_length=512, truncation=True,
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

# ----------------- SHAP EXPLANATION -----------------


def explain_model(model, input, aspect):
    def get_aspect_explanation(input):
        IE, NS, TF, JP = model.getBinaryProbs(np.array(input))

        if aspect == 'IE':
            return IE.numpy()
        elif aspect == 'NS':
            return NS.numpy()
        elif aspect == 'TF':
            return TF.numpy()
        elif aspect == 'JP':
            return JP.numpy()
        else:
            raise ValueError("Invalid aspect provided.")
    class_names = get_class_names(aspect)
    aspect_explainer = LimeTextExplainer(class_names=class_names)
    if isinstance(input, str):
        input = clean_text_for_explain(input)
        return aspect_explainer.explain_instance(input, get_aspect_explanation, num_features=500)
    else:
        input = [clean_text_for_explain(text) for text in input]
        return [aspect_explainer.explain_instance(text, get_aspect_explanation, num_features=500) for text in input]

# ----------------- HELPER FUNCTIONS -----------------


def get_class_names(aspect):
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


# Text Cleaning Functions
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))


def mystopwords(text):
    return ' '.join([w for w in word_tokenize(text) if not w in stop_words])


def clean_text(string):
    clean = re.sub(r"(?:\@|http?\://|https?\://|www)\S+|\#\w+",
                   "", string)  # remove mentions & hashtags
    # remove url
    clean = re.sub(
        r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', ' ', clean)
    clean = re.sub('[\n]', ' ', clean)  # remove newline character
    # remove non alphabetic characters
    clean = re.sub('[^a-zA-Z]', ' ', clean.lower())
    clean = re.sub(r'[,]', ' ', clean)
    clean = mystopwords(clean)  # remove stopwords
    clean = re.sub(r'\s+', ' ', clean)  # removve extra spaces
    return clean


def clean_text_for_explain(string):
    clean = re.sub(r"(?:\@|http?\://|https?\://|www)\S+|\#\w+",
                   "", string)  # remove mentions & hashtags
    clean = re.sub(r'\s+', ' ', clean)  # removve extra spaces
    return clean
