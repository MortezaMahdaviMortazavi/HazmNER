import torch
import torch.nn as nn
import torch.nn.functional as F
import config
from transformers import BertModel


class Bert(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(config.MODEL_NAME)
        self.bert_drop = nn.Dropout(0.2)
        self.classifier = nn.Linear(self.bert.config.hidden_size, config.NUM_CLASSES)
        
    def forward(self,ids, attention_mask, token_type_ids):
        bert_output = self.bert(ids,attention_mask,token_type_ids)[0]
        bert_output = self.bert_drop(bert_output)
        return self.classifier(bert_output)
    


        