import numpy as np
import torch
from transformers import AutoModel


PRE_TRAINED_MODEL_NAME = 'roberta-base'

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
    def forward(self, input_ids, attn_mask, token_type_ids):
        output = self.bert_model(
            input_ids, 
            attention_mask=attn_mask, 
            token_type_ids=token_type_ids
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
    def getProbabilities(self, output):
        # apply sigmoid function
        probabilities=torch.sigmoid(output)
        return probabilities
    def getPrediction(self, output):
        # apply sigmoid function and round the result
        prediction=torch.sigmoid(output)
        prediction=torch.round(prediction)
        return prediction