import torch
import torch.nn as nn
import transformers 
import torch.nn.functional as F

import math

"""
Contents:
    Modules:
        1. CrossAttention
        2. ConceptTransformer
    Model:    
        3. EnterpreneurConceptTransformer
    Training Helpers:
        4. pre_trained_tokenizer
        5. explanation_loss
        6. train_epoch
        7. eval_model
        8. get_predictions_with_concepts
"""


"-------------------------------Modules: `CrossAttention` and `ConceptTransformer`------------------------------------"

class CrossAttention(nn.Module):
    """
    Obtained from: https://github.com/IBM/concept_transformer

    A class to represent the CrossAttention module.
    Input:
        x: Query sequence, transformed using a linear layer (q).
        y: Key and value sequence, transformed using a linear layer (kv).
    Process:
        Computes attention scores between queries and keys.
        Applies softmax and dropout for regularization.
        Combines values based on attention scores.
    """

    def __init__(
        self, dim, n_outputs=None, num_heads=8, attention_dropout=0.1, projection_dropout=0.0
    ):
        super().__init__()
        n_outputs = n_outputs if n_outputs else dim
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, dim * 2, bias=False)
        self.attn_drop = nn.Dropout(attention_dropout)

        self.proj = nn.Linear(dim, n_outputs)
        self.proj_drop = nn.Dropout(projection_dropout)

    def forward(self, x, y):
        B, Nx, C = x.shape
        By, Ny, Cy = y.shape

        assert C == Cy, "Feature size of x and y must be the same"

        q = self.q(x).reshape(B, Nx, 1, self.num_heads, C //
                              self.num_heads).permute(2, 0, 3, 1, 4)
        kv = (
            self.kv(y)
            .reshape(By, Ny, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )

        q = q[0]
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nx, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn
    
class ConceptTransformer(nn.Module):
    """
    Obtained from: https://github.com/IBM/concept_transformer

    A class to represent the ConceptTransformer module.
    Components:
        Utilizes cross-attention mechanism.
        Represents unsupervised and supervised concepts.
        Performs sequence pooling for aggregated representations.
    Initialization:
        Initializes concept parameters and creates cross-attention instances.
    Forward Pass:
        Computes token attention and performs sequence pooling.
        Optionally applies cross-attention with concepts.
        Aggregates outputs and returns final results with attention scores.
    """
    def __init__(self, dim, num_classes, num_heads=8, attention_dropout=0.1, projection_dropout=0.0,n_unsup_concepts=0,n_concepts=5):
        super().__init__()
        self.cross_attention = CrossAttention(
            dim, num_classes, num_heads, attention_dropout, projection_dropout
        )
        # Unsupervised concepts
        self.n_unsup_concepts = n_unsup_concepts
        self.unsup_concepts = nn.Parameter(
            torch.zeros(1, n_unsup_concepts, dim), requires_grad=True
        )
        nn.init.trunc_normal_(self.unsup_concepts,
                              std=1.0 / math.sqrt(dim))
        if n_unsup_concepts > 0:
            self.unsup_cross_attention = CrossAttention(
                dim=dim,
                n_outputs=num_classes,
                num_heads=num_heads,
                attention_dropout=attention_dropout,
                projection_dropout=projection_dropout,
            )

        # Supervised concepts
        self.n_concepts = n_concepts
        self.concepts = nn.Parameter(torch.zeros(
            1, n_concepts, dim), requires_grad=True)
        nn.init.trunc_normal_(self.concepts, std=1.0 /
                              math.sqrt(dim))
        if n_concepts > 0:
            self.concept_cross_attention = CrossAttention(
                dim=dim,
                n_outputs=num_classes,
                num_heads=num_heads,
                attention_dropout=attention_dropout,
                projection_dropout=projection_dropout,
            )
         # Sequence pooling for both supervised and unsupervised concepts
        if n_concepts > 0 or n_unsup_concepts > 0:
            self.token_attention_pool = nn.Linear(dim, 1)

    def forward(self, x):
        unsup_concept_attn, concept_attn = None, None

        out = 0
        if self.n_unsup_concepts > 0 or self.n_concepts > 0:
            token_attn = torch.softmax(
                self.token_attention_pool(x), dim=1).transpose(-1, -2)
            x_pooled = torch.matmul(token_attn, x)

        if self.n_unsup_concepts > 0:  # unsupervised stream
            out_unsup, unsup_concept_attn = self.unsup_cross_attention(
                x_pooled, self.unsup_concepts)
            unsup_concept_attn = unsup_concept_attn.mean(
                1)  # average over heads
            out = out + out_unsup.squeeze(1)  # squeeze token dimension

        if self.n_concepts > 0:  # supervised stream
            out_n, concept_attn = self.concept_cross_attention(
                x_pooled, self.concepts)
            concept_attn = concept_attn.mean(1)  # average over heads
            out = out + out_n.squeeze(1)  # squeeze token dimension

        return out, unsup_concept_attn, concept_attn

    
"-------------------------------Model: `ConceptEnterpreneurClassifier`------------------------------------"

class EnterpreneurConceptTransformer(nn.Module):
    """
    A class to represent the EnterpreneurConceptTransformer model using BERT, Dropout and Linear layers for classification.
    CONCEPTS INVOLVED HERE
    ----------------------------------------------------------------------------------------------------------
    """
    def __init__(self, n_classes, n_unsup_concepts=0, n_concepts=5,PRE_TRAINED_MODEL_NAME='bert-base-cased'):
        super(EnterpreneurConceptTransformer, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.concept_transformer = ConceptTransformer(
            dim=self.bert.config.hidden_size,
            num_classes=n_classes,
            n_unsup_concepts=n_unsup_concepts,
            n_concepts=n_concepts,
        )
        self.softmax = F.log_softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask)
        
        concept_output, unsup_concept_attn, concept_attn = self.concept_transformer(
            bert_output.last_hidden_state)
        
        output = self.softmax(concept_output)

        return output, concept_attn, unsup_concept_attn

"-------------------------------Training Helpers: `pre_trained_tokenizer`, `explanation_loss`, `train_epoch`, `eval_model`, `get_predictions_with_concepts`------------------------------------"

def pre_trained_tokenizer(PRE_TRAINED_MODEL_NAME):
    tokenizer = transformers.BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    return tokenizer

def explanation_loss(concept_attn, attn_targets):
    """
    Obtained from: https://github.com/IBM/concept_transformer

        Supervised concept loss using MSE.
        Attention targets are normalized to sum to 1,
        but cost is normalized to be O(1)

    Args:
        attn_targets, torch.tensor of size (batch_size, n_concepts): one-hot attention targets
    """
    if concept_attn is None:
        return 0.0
    if attn_targets.dim() < 3:
        attn_targets = attn_targets.unsqueeze(1)
    norm = attn_targets.sum(-1, keepdims=True)
    idx = ~torch.isnan(norm).squeeze()
    if not torch.any(idx):
        return 0.0
    # MSE requires both floats to be of the same type
    norm_attn_targets = (attn_targets[idx] / norm[idx]).float()
    n_concepts = norm_attn_targets.shape[-1]
    return n_concepts * F.mse_loss(concept_attn[idx], norm_attn_targets, reduction="mean")

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    pass

def eval_model(model, data_loader, loss_fn, device, n_examples):
    pass

def get_predictions_with_concepts(model, data_loader, device):
    pass
