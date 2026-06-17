# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict
import pdb
import numpy as np

import torch
import torch.nn.functional as F


from torch import nn
from typing import Dict, List

from utils.misc import NestedTensor, is_main_process
# from .position_encoding import build_position_encoding

#from pytorch_pretrained_bert.modeling import BertModel
#from transformers import BertModel
from transformers import AutoModel, BertForMaskedLM, BertConfig, AutoTokenizer, AutoConfig

'''
class BERT(nn.Module):
    def __init__(self, name: str, train_bert: bool, hidden_dim: int, max_len: int, enc_num):
        super().__init__()
        if 'bert-base-uncased' in name:
            self.num_channels = 768
        else:
            self.num_channels = 1024
        self.enc_num = enc_num
        pdb.set_trace()
        self.bert = BertModel.from_pretrained(name)

        if not train_bert:
            for parameter in self.bert.parameters():
                parameter.requires_grad_(False)

    def forward(self, tensor_list: NestedTensor):
        #pdb.set_trace()
        if self.enc_num > 0:
            xs = self.bert(tensor_list.tensors, token_type_ids=None, attention_mask=tensor_list.mask).last_hidden_state
        else:
            xs = self.bert.embeddings.word_embeddings(tensor_list.tensors)

        mask = tensor_list.mask.to(torch.bool)
        mask = ~mask
        out = NestedTensor(xs, mask)

        return out

'''
from .utils.nn import (
    SiLU,
    linear,
    timestep_embedding,
)
class BERT(nn.Module):
    def __init__(self, name: str, train_bert: bool, hidden_dim: int, max_len: int, enc_num):
        super().__init__()
        config = AutoConfig.from_pretrained(name)
        self.num_channels = config.hidden_size
        self.enc_num = enc_num
        #pdb.set_trace()
        self.bert = AutoModel.from_pretrained(name)
        # Build a transform head with dimensions matched to the loaded BERT family checkpoint.
        bert_mlm = BertForMaskedLM(BertConfig.from_pretrained(name))
        self.mlp = bert_mlm.cls.predictions.transform
        self.tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
        #pdb.set_trace()
        #pos = self.bert.positional_embedding
        #diffusion_process      
        # self.input_dims = 128
        # self.hidden_t_dim = 128
        # self.output_dims = 128
        # self.dropout = 0.1
        # self.logits_mode = 1
        # config = AutoConfig.from_pretrained(name)
        # config.hidden_dropout_prob = self.dropout
        # self.hidden_size = config.hidden_size
        # time_embed_dim = self.hidden_t_dim * 4
        # vocab_size = 30522
        # #pdb.set_trace()
        # self.time_embed = nn.Sequential(
        #     linear(self.hidden_t_dim, time_embed_dim),
        #     SiLU(),
        #     linear(time_embed_dim, config.hidden_size),
        # )
        # self.word_embedding = nn.Embedding(vocab_size, self.input_dims)
        # self.lm_head = nn.Linear(self.input_dims, vocab_size)
        # with torch.no_grad():
        #     self.lm_head.weight = self.word_embedding.weight

        # if self.input_dims != config.hidden_size:
        #     self.input_up_proj = nn.Sequential(nn.Linear(self.input_dims, config.hidden_size),
        #                                       nn.Tanh(), nn.Linear(config.hidden_size, config.hidden_size))
        
        # #if init_pretrained == 'sup-roberta':
        #     #pdb.set_trace()
        # print('initializing from pretrained sup-roberta...')
        # #pdb.set_trace()
        # #print(config)
        # #temp_bert = BertModel.from_pretrained(config_name, config=config)
        # #temp_bert = AutoModel.from_pretrained(config_name, config=config)
        # self.word_embedding = self.bert.embeddings.word_embeddings
        # with torch.no_grad():
        #     self.lm_head.weight = self.word_embedding.weight
        # # self.lm_head.weight.requires_grad = False
        # # self.word_embedding.weight.requires_grad = False
        
        # self.input_transformers = self.bert.encoder
        # self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        # self.position_embeddings = self.bert.embeddings.position_embeddings
        # self.LayerNorm = self.bert.embeddings.LayerNorm

        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # #del self.bert.embeddings
        # #del self.bert.pooler
        # if self.output_dims != config.hidden_size:
        #     self.output_down_proj = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
        #                                         nn.Tanh(), nn.Linear(config.hidden_size, self.output_dims))


        # self.output_trans = nn.Linear(self.input_dims, self.num_channels)

        # if not train_bert:
        #     for parameter in self.bert.parameters():
        #         parameter.requires_grad_(False)

        
    '''
    def forward(self, tensor_list: NestedTensor):
        #pdb.set_trace()
        if self.enc_num > 0:
            xs = self.bert(tensor_list.tensors, token_type_ids=None, attention_mask=tensor_list.mask).last_hidden_state
        else:
            xs = self.bert.embeddings.word_embeddings(tensor_list.tensors)

        mask = tensor_list.mask.to(torch.bool)
        mask = ~mask
        out = NestedTensor(xs, mask)

        return out

    '''


    def get_delta(self, device, template = '*cls*_This_sentence_of_"*sent_0*"_means*mask*.*sep+*'):
        #pdb.set_trace()
        template = template.replace('*mask*', self.tokenizer.mask_token)\
                    .replace('*sep+*', '')\
                    .replace('*cls*', '').replace('*sent_0*', ' ')
        # strip for roberta tokenizer
        bs_length = len(self.tokenizer.encode(template.split(' ')[0].replace('_', ' ').strip())) - 2 + 1
        # replace for roberta tokenizer
        batch = self.tokenizer([template.replace('_', ' ').strip().replace('   ', ' ')], return_tensors='pt')
        #pdb.set_trace()
        batch['position_ids'] = torch.arange(batch['input_ids'].shape[1]).to(device).unsqueeze(0)
        for k in batch:
            batch[k] = batch[k].repeat(256, 1).to(device)
        batch['position_ids'][:, bs_length:] += torch.arange(256).to(device).unsqueeze(-1)
        m_mask = batch['input_ids'] == self.tokenizer.mask_token_id
        #pdb.set_trace()
        outputs = self.bert(**batch,  output_hidden_states=True, return_dict=True)
        last_hidden = outputs.hidden_states[-1]
        delta = last_hidden[m_mask]
        #delta.requires_grad = False
        #import pdb;pdb.set_trace()
        template_len = batch['input_ids'].shape[1]
        return delta, template_len



    #def forward(self, text_info, text_info_s, tensor_list: NestedTensor, tensor_list_s: NestedTensor, device):
    def forward(self, text_info, tensor_list: NestedTensor, word_selection, device):    
        #sentence embedding
        delta, template_len = self.get_delta(device)
        outputs = self.bert(**text_info, output_hidden_states=True, return_dict=True)
        #valid_expression_features
        #pdb.set_trace()
        text_length = text_info['attention_mask'].size(1)
        valid_text_mask = text_info['attention_mask'] * word_selection[:,0:text_length]
        valid_text_info = {}
        valid_text_info['input_ids'] = text_info['input_ids']
        valid_text_info['attention_mask'] = valid_text_mask
        valid_outputs = self.bert(**valid_text_info, output_hidden_states=True, return_dict=True)
        #pdb.set_trace()
        #outputs_s = self.bert(**text_info_s, output_hidden_states=True, return_dict=True)
        try:
            pooler_output = outputs.pooler_output
            #pooler_output_s = outputs_s.pooler_output
            valid_pooler_output = valid_outputs.pooler_output
        except AttributeError:
            pooler_output = outputs['last_hidden_state'][:, 0, :]
            #pooler_output_s = outputs_s['last_hidden_state'][:, 0, :]
            valid_pooler_output = valid_outputs['last_hidden_state'][:, 0, :]

        #sentences    
        last_hidden = outputs.last_hidden_state
        pooler_output = last_hidden[text_info['input_ids'] == self.tokenizer.mask_token_id]
        pooler_output = self.mlp(pooler_output)
        blen = text_info['attention_mask'].sum(-1) - template_len
        pooler_output -= self.mlp(delta[blen])
        sentences_out = pooler_output.view(text_info['input_ids'].shape[0], -1)
        #valid sentences
        valid_last_hidden = valid_outputs.last_hidden_state
        valid_pooler_output = valid_last_hidden[text_info['input_ids'] == self.tokenizer.mask_token_id]
        valid_pooler_output = self.mlp(valid_pooler_output)
        valid_blen = valid_text_info['attention_mask'].sum(-1) - template_len
        valid_pooler_output -= self.mlp(delta[valid_blen])
        valid_sentences_out = valid_pooler_output.view(valid_text_info['input_ids'].shape[0], -1)


        #sentence word embedding
        if self.enc_num > 0:
            xs = self.bert(tensor_list.tensors, token_type_ids=None, attention_mask=tensor_list.mask).last_hidden_state
        else:
            xs = self.bert.embeddings.word_embeddings(tensor_list.tensors)
        
        mask = tensor_list.mask.to(torch.bool)
        mask = ~mask
        out = NestedTensor(xs, mask)
        #similar sentence word embedding
        # if self.enc_num > 0:
        #     xs_s = self.bert(tensor_list_s.tensors, token_type_ids=None, attention_mask=tensor_list_s.mask).last_hidden_state
        # else:
        #     xs_s = self.bert.embeddings.word_embeddings(tensor_list_s.tensors)

        # mask_s = tensor_list_s.mask.to(torch.bool)
        # mask_s = ~mask_s
        # out_s = NestedTensor(xs_s, mask_s)
        #return out, out_s, sentences_out, sentences_out_s
        return out, sentences_out, valid_sentences_out

    def get_embeds(self, input_ids):
        return self.word_embedding(input_ids)

    def get_logits(self, hidden_repr):
        if self.logits_mode == 1:
            return self.lm_head(hidden_repr)
        elif self.logits_mode == 2: # standard cosine similarity
            text_emb = hidden_repr
            emb_norm = (self.lm_head.weight ** 2).sum(-1).view(-1, 1)  # vocab
            text_emb_t = torch.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1)  # d, bsz*seqlen
            arr_norm = (text_emb ** 2).sum(-1).view(-1, 1)  # bsz*seqlen, 1
            dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * torch.mm(self.lm_head.weight,
                                                                     text_emb_t)  # (vocab, d) x (d, bsz*seqlen)
            scores = torch.sqrt(torch.clamp(dist, 0.0, np.inf)).view(emb_norm.size(0), hidden_repr.size(0),
                                                               hidden_repr.size(1)) # vocab, bsz*seqlen
            scores = -scores.permute(1, 2, 0).contiguous()
            return scores
        else:
            raise NotImplementedError


    def language_trans(self, x, timesteps):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x C x ...] Tensor of outputs.
        """
        #pdb.set_trace()
        emb_t = self.time_embed(timestep_embedding(timesteps, self.hidden_t_dim))
        '''
        if self.input_dims != self.hidden_size:
            emb_x = self.input_up_proj(x)
        else:
        '''
        emb_x = x
        seq_length = x.size(1)
        #pdb.set_trace()
        position_ids = self.position_ids[:, : seq_length ]
        # print(emb_x.shape, emb_t.shape, self.position_embeddings)
        emb_inputs = self.position_embeddings(position_ids) + emb_x + emb_t.unsqueeze(1).expand(-1, seq_length, -1)
        emb_inputs = self.dropout(self.LayerNorm(emb_inputs))

        input_trans_hidden_states = self.input_transformers(emb_inputs).last_hidden_state
        
        if self.output_dims != self.hidden_size:
            h = self.output_down_proj(input_trans_hidden_states)
        else:
            h = input_trans_hidden_states
        h = h.type(x.dtype)
        #pdb.set_trace()
        return self.output_trans(h)


        
        #pdb.set_trace()
        if self.enc_num > 0:
            xs = self.bert_a(text_info).last_hidden_state
        else:
            xs = self.bert.embeddings.word_embeddings(text_info)

        #mask = tensor_list.mask.to(torch.bool)
        #mask = ~mask
        #out = NestedTensor(xs, mask)

        return out
    











def build_bert(args):
    # position_embedding = build_position_encoding(args)
    train_bert = args.lr_bert > 0
    #pdb.set_trace()
    bert = BERT(args.bert_model, train_bert, args.hidden_dim, args.max_query_len, args.bert_enc_num)
    # model = Joiner(bert, position_embedding)
    # model.num_channels = bert.num_channels
    return bert
