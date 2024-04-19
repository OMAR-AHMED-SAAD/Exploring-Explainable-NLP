import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

PRE_TRAINED_MODEL_NAME = 'roberta-base'
MAX_LEN = 512
tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(99)
torch.cuda.manual_seed(99)
torch.cuda.manual_seed_all(99)
np.random.seed(99)


class ROBERTAClass(torch.nn.Module):
    def __init__(self, PRE_TRAINED_MODEL_NAME, num_classes=4, dropout=0.3):
        super(ROBERTAClass, self).__init__()
        self.bert_model = AutoModel.from_pretrained(PRE_TRAINED_MODEL_NAME, return_dict=True, output_attentions=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(768, num_classes)
    def forward(self, input):
        output = self.bert_model(
           **input # input_ids, attention_mask, token_type_ids
        )
        output_dropout = self.dropout(output.pooler_output)
        output = self.linear(output_dropout)
        return output
    def getAttention(self, input_ids, attn_mask, token_type_ids):
        output = self.bert_model(
            input_ids, 
            attention_mask=attn_mask, 
            token_type_ids=token_type_ids
        )
        return output.attentions
    def getPrediction(self, input):
        # if input is list of strings
        if isinstance(input, list):
            input = [clean_text(text) for text in input]
        elif isinstance(input, str):
            input = clean_text(input) 
        input = tokenizer(input, max_length=MAX_LEN, truncation=True,
                          padding='longest', return_token_type_ids=True, return_tensors='pt')
        input = {key: tensor.to(device) for key, tensor in input.items()}
        output = self.forward(input)
        probabilities = torch.sigmoid(output)
        return probabilities
    

#-----------------Helper Functions-----------------#

# Text Cleaning Functions
import nltk
from nltk import word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('punkt')
stop_words = set(stopwords.words('english'))


def mystopwords(text):
    return ' '.join([w for w in word_tokenize(text) if not w in stop_words])

import re
def clean_text(string):
    clean = re.sub(r"(?:\@|http?\://|https?\://|www)\S+|\#\w+", "", string) # remove mentions, hashtags
    clean=re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', ' ',clean) #remove url
    clean=re.sub('[\n]',' ',clean) #remove newline character
    clean=re.sub('[^a-zA-Z]',' ',clean.lower()) #remove non alphabetic characters
    clean=re.sub(r'[,]', ' ', clean)
    clean=mystopwords(clean) #remove stopwords
    clean = re.sub(r'\s+', ' ', clean)
    return clean