import torch.nn as nn
import transformers 

class EnterpreneurClassifier(nn.Module):
    """
    A class to represent the EnterpreneurClassifier model using BERT, Dropout and Linear layers for classification.
    NO CONCEPTS INVOLVED HERE
    ----------------------------------------------------------------------------------------------------------
    """
    def __init__(self, n_classes,PRE_TRAINED_MODEL_NAME):
        super(EnterpreneurClassifier, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask)
        output = self.drop(bert_output.pooler_output)
        output = self.out(output)
        return self.softmax(output) 
        