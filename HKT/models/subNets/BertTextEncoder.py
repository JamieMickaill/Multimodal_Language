import os
import sys
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertTokenizer, BertModel,BertForSequenceClassification

__all__ = ['BertTextEncoder']

class BertTextEncoder(nn.Module):
    def __init__(self, language='en', use_finetune=False):
        """
        language: en / cn
        """
        super(BertTextEncoder, self).__init__()

        assert language in ['en', 'cn']

        tokenizer_class = BertTokenizer
        model_class = BertForSequenceClassification
        # directory is fine
        # pretrained_weights = '/home/sharing/disk3/pretrained_embedding/Chinese/bert/pytorch'
        if language == 'en':
            self.tokenizer = tokenizer_class.from_pretrained('bert-base-uncased', do_lower_case=True)
            self.model = model_class.from_pretrained('bert-base-uncased',num_labels=1)
        # elif language == 'cn':
        #     self.tokenizer = tokenizer_class.from_pretrained('pretrained_model/bert_cn')
        #     self.model = model_class.from_pretrained('pretrained_model/bert_cn')
        
        self.use_finetune = use_finetune
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def from_text(self, text):
        """
        text: raw data
        """
        input_ids = self.get_id(text)
        with torch.no_grad():
            last_hidden_states = self.model(input_ids)[0]  # Models outputs are now tuples
        return last_hidden_states.squeeze()
    
    def forward(self, text):
        """
        text: (batch_size, 3, seq_len)
        3: input_ids, input_mask, segment_ids
        input_ids: input_ids,
        input_mask: attention_mask,
        segment_ids: token_type_ids
        """
        input_ids, input_mask, segment_ids = text[:,0,:].long(), text[:,1,:].float(), text[:,2,:].long()
        if self.use_finetune:
            last_hidden_states = self.model(input_ids=input_ids,
                                            attention_mask=input_mask,
                                            token_type_ids=segment_ids)[0]  # Models outputs are now tuples
        else:
            with torch.no_grad():
                last_hidden_states = self.model(input_ids=input_ids,
                                                attention_mask=input_mask,
                                                token_type_ids=segment_ids)[0]  # Models outputs are now tuples
        return last_hidden_states



class BertTextEncoderRegressionHead(nn.Module):
    def __init__(self, language='en', use_finetune=False):
        """
        language: en / cn
        """
        super(BertTextEncoderRegressionHead, self).__init__()

        assert language in ['en', 'cn']

        tokenizer_class = BertTokenizer
        model_class = BertModel
        # directory is fine
        # pretrained_weights = '/home/sharing/disk3/pretrained_embedding/Chinese/bert/pytorch'
        if language == 'en':
            self.tokenizer = tokenizer_class.from_pretrained('bert-base-uncased')
            self.model = model_class.from_pretrained('bert-base-uncased')
        # elif language == 'cn':
        #     self.tokenizer = tokenizer_class.from_pretrained('pretrained_model/bert_cn')
        #     self.model = model_class.from_pretrained('pretrained_model/bert_cn')
        
        self.use_finetune = use_finetune
        self.regression_head = nn.Linear(768, 1) 
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def from_text(self, text):
        """
        text: raw data
        """
        input_ids = self.get_id(text)
        with torch.no_grad():
            last_hidden_states = self.model(input_ids)[0]  # Models outputs are now tuples
        return last_hidden_states.squeeze()
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        """
        text: (batch_size, 3, seq_len)
        3: input_ids, input_mask, segment_ids
        input_ids: input_ids,
        input_mask: attention_mask,
        segment_ids: token_type_ids
        """
        # input_ids, input_mask, segment_ids = text[:,0,:].long(), text[:,1,:].float(), text[:,2,:].long()
        if self.use_finetune:
            last_hidden_states = self.model(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            token_type_ids=token_type_ids)[0]  # Models outputs are now tuples
        else:
            with torch.no_grad():
                last_hidden_states = self.model(input_ids=input_ids,
                                                attention_mask=attention_mask,
                                            token_type_ids=token_type_ids)[0]  # Models outputs are now tuples
        cls_embedding = last_hidden_states[:, 0, :]  # Taking the [CLS] embedding
        regression_output = self.regression_head(cls_embedding)  # (batch_size,)
        return (regression_output,cls_embedding,last_hidden_states)
    
if __name__ == "__main__":
    bert_normal = BertTextEncoderRegressionHead()
