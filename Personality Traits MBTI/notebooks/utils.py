import json
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
import torch

# save the training history to a json file

def saveHistory(history):
    with open('logs/history.json', 'w') as f:
        json.dump(history, f)


# print the training history
def printHistory(history):
    if history is not None:
        for i in range(len(history['train_acc'])):
            print(f'Epoch {i+1}/{len(history["train_acc"])}')
            print(f'train_loss={history["train_loss"][i]:.4f}, val_loss={history["val_loss"][i]:.4f} train_acc={history["train_acc"][i]:.4f}, val_acc={history["val_acc"][i]:.4f}')
            print('-----------------------------------------')  

# plot the training history for the loss and accuracy

def plot_training_history(history):
    epochs = range(1, len(history['train_acc']) + 1)  # Assuming train_acc and val_acc have the same length
    plt.plot(epochs, history['train_acc'], label='Train Accuracy')
    plt.plot(epochs, history['val_acc'], label='Validation Accuracy')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
     # Set the x-axis ticks to integers
    plt.xticks(range(1, len(epochs) + 1))
    plt.legend()
    plt.show()

    plt.plot(epochs,history['train_loss'], label='train loss')
    plt.plot(epochs,history['val_loss'], label='validation loss')
    plt.title('Training history')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
     # Set the x-axis ticks to integers
    plt.xticks(range(1, len(epochs) + 1))
    plt.ylim([0,2])
    plt.show()        

# get the predictions
def get_predictions(model,data_loader,device):
    model.eval()
    text=[]
    predictions = []
    predictions_probs = []
    real_values = []
    with torch.no_grad():
        for data in data_loader:
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


def get_metrics(model,test_data_loader,target_list,device):
    # get predictions for the test data
    text, predictions, prediction_probs, real_values = get_predictions(model, test_data_loader,device)
    accuracy = accuracy_score(real_values.view(-1), predictions.view(-1))
    accuracy_scores = {}
    for i in range(len(target_list)):
        accuracy_scores[target_list[i]] = accuracy_score(real_values[:,i], predictions[:,i])
    report=classification_report(real_values, predictions, target_names=target_list)
    print(f"Accuracy {accuracy}")
    print(f'Accuracy Scores\n{accuracy_scores}')
    print(f"classification_report\n{report}")
    # return accuracy,accuracy_score,report