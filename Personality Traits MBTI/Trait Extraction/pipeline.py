import numpy as np
import os
import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

# import model
import sys
sys.path.append('../Models')
import MBTI_model_shap as model

# setting the random seed
torch.manual_seed(99)
torch.cuda.manual_seed(99)
torch.cuda.manual_seed_all(99)
np.random.seed(99)

# model settings
PRE_TRAINED_MODEL_NAME = 'roberta-base'
tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if os.path.exists('ckpts'):
    CHECKPOINTPATH = 'ckpts/Persnality_MBTI'
else:
    CHECKPOINTPATH = '../ckpts/Persnality_MBTI'

# load the model
roberta_model = model.ROBERTAClass(PRE_TRAINED_MODEL_NAME)
roberta_model.load_state_dict(torch.load(
    CHECKPOINTPATH + f'_clean_Best_{PRE_TRAINED_MODEL_NAME}.bin', map_location=torch.device(device)))
roberta_model.to(device)

# create the dataset class
class TweetsDataset(Dataset):
    def __init__(self, data, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data.text[index]
        text = clean_text(text)
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
        }

# create the data loaders function
def getDataLoaders(data, batch_size):
    # get the max length of the text in tokens
    max_len = max([len(tokenizer.encode(text)) for text in data.text])
    # create the dataset
    dataset = TweetsDataset(data, max_len)
    # create the data loaders
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader


def get_dataframe(datapath, text_column):
    data = pd.read_csv(datapath)
    # rename the text column to 'text'
    data.rename(columns={text_column: 'text'}, inplace=True)
    data = data[['text']]
    # drop the null values
    data.dropna(inplace=True)
    # remove the duplicates
    data.drop_duplicates(inplace=True)
    # remove the tweets with less than 8 words
    data = data[data['text'].apply(lambda x: len(x.split()) > 8)]
    # reset the index
    data.reset_index(drop=True, inplace=True)
    # print the text column count
    print("dataset size after cleaning:", data['text'].count())
    return data

from tqdm import tqdm

def get_predictions(model, data_loader):
    model.eval()
    predictions = []
    with torch.no_grad(), tqdm(total=len(data_loader)) as pbar:
        for data in data_loader:
            ids = data['input_ids'].to(device, dtype=torch.long)
            mask = data['attention_mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(
                device, dtype=torch.long)
            outputs = model(ids, mask, token_type_ids)
            outputs = torch.sigmoid(outputs)
            predictions.extend(outputs.cpu().detach().round())
            pbar.update(1)  # Update progress bar
    predictions = torch.stack(predictions)
    return predictions


# -----------------Helper Functions-----------------#


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
    clean = re.sub(r"(?:\@|http?\://|https?\://|www)\S+|\#\w+",
                   "", string)  # remove mentions, hashtags
    # remove url
    clean = re.sub(
        r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', ' ', clean)
    clean = re.sub('[\n]', ' ', clean)  # remove newline character
    # remove non alphabetic characters
    clean = re.sub('[^a-zA-Z]', ' ', clean.lower())
    clean = re.sub(r'[,]', ' ', clean)
    clean = mystopwords(clean)  # remove stopwords
    clean = re.sub(r'\s+', ' ', clean)
    return clean

# measure the presence of traits
def measure_traits_presence(predictions, print_out=True):
    # Count the number of ones (1s) along each column (dimension 0)
    ones_count = torch.sum(predictions, dim=0)
    # Count the number of zeros (0s) along each column (dimension 0)
    zeros_count = predictions.size(0) - ones_count
    # Calculate the percentage of ones and zeros for each trait
    ones_precentage = ones_count/predictions.size(0)
    zeros_precentage = zeros_count/predictions.size(0)
    # Round the percentage to 2 decimal places and multiply by 100 to get the percentage
    ones_precentage = (torch.round(ones_precentage, decimals=2)*100).round()
    zeros_precentage = (torch.round(zeros_precentage, decimals=2)*100).round()
    if print_out:
        print("Introversion(0) vs Extroversion(1):", zeros_precentage[0].item(
        ), "% vs", ones_precentage[0].item(), "%")
        print("Intuition(0) vs Sensing(1):", zeros_precentage[1].item(
        ), "% vs", ones_precentage[1].item(), "%")
        print("Thinking(0) vs Feeling(1):", zeros_precentage[2].item(
        ), "% vs", ones_precentage[2].item(), "%")
        print("Judging(0) vs Perceiving(1):", zeros_precentage[3].item(
        ), "% vs", ones_precentage[3].item(), "%")

    return ones_precentage, zeros_precentage


# main function
def get_enterpreneur_traits(data_path, text_column, entrepreneur):
    # print the entrepreneur name
    print(f"Analyzing {entrepreneur} tweets")
    # get the data
    data = get_dataframe(data_path, text_column)
    # get the data loaders
    data_loader = getDataLoaders(data, batch_size=32)
    # get the predictions
    predictions = get_predictions(roberta_model, data_loader)
    # measure the presence of traits
    ones_precentage, zeros_precentage = measure_traits_presence(predictions)
    # save the percentage of traits in json file
    with open(f'{entrepreneur} traits.json', 'w') as f:
        json.dump({"Introversion": zeros_precentage[0].item(), "Extroversion": ones_precentage[0].item(),
                   "Intuition": zeros_precentage[1].item(), "Sensing": ones_precentage[1].item(),
                   "Thinking": zeros_precentage[2].item(), "Feeling": ones_precentage[2].item(),
                   "Judging": zeros_precentage[3].item(), "Perceiving": ones_precentage[3].item()}, f)
