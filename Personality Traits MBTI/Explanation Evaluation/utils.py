import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import torch


# delete words (personality traits) from the text
words=["intj","intp","entj","entp","infj","infp","enfj","enfp","istj","isfj","estj","esfj","istp","isfp","estp","esfp", "introvert", "extrovert", "intuitive", "observant", "thinking", "feeling", "judging", "preceiving"]

def delete_words(data):
    for word in words:
        data['posts'] = data['posts'].str.replace(word, '')
    return data

# get the predictions
def get_predictions(model,data_loader,device):
    model.eval()
    text=[]
    predictions = []
    predictions_probs = []
    real_values = []
    i=0
    with torch.no_grad():
        for data in data_loader:
            # print the data shape
            if i==0:
                print(f'input_ids shape: {data["input_ids"].shape}')
                print(f'attention_mask shape: {data["attention_mask"].shape}')
                print(f'token_type_ids shape: {data["token_type_ids"].shape}')
                print(f'targets shape: {data["targets"].shape}')
            print(i)
            i+=1
            text.extend(data['text'])
            ids = data['input_ids'].to(device, dtype = torch.long)
            mask = data['attention_mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model(ids, mask, token_type_ids)
            outputs = torch.sigmoid(outputs)
            predictions.extend(outputs.cpu().detach().round())
            real_values.extend(targets.cpu().detach())
            predictions_probs.extend(outputs.cpu().detach())      
    predictions = torch.stack(predictions)
    predictions_probs = torch.stack(predictions_probs)
    real_values = torch.stack(real_values)
    return text,predictions,predictions_probs, real_values    


# def get_accuracy(model,test_data_loader,target_list,device,aspect):
#     # get predictions for the test data
#     text, predictions, prediction_probs, real_values = get_predictions(model, test_data_loader,device)
#     accuracy_scores = {}
#     for i in range(len(target_list)):
#         accuracy_scores[target_list[i]] = accuracy_score(real_values[:,i], predictions[:,i])
#     print(f'Accuracy Scores\n{accuracy_scores[aspect]}')

# get the accuracy
def get_accuracy(model, y_true, y_pred,target_list):
    accuracy_scores = {}
    for i in range(len(target_list)):
        accuracy_scores[target_list[i]] = accuracy_score(y_true[:,i], y_pred[:,i])
    print(f'Accuracy Scores\n{accuracy_scores}')
    return accuracy_scores

#  get correct predictions
def get_correct_predictions(predictions, real_values):
    correct_predictions = []
    for i in range(len(real_values)):
        if (predictions[i] == real_values[i]).sum() == len(real_values[i]):
            correct_predictions.append(i)
    return correct_predictions

