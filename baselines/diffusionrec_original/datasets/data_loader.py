# -*- coding: utf-8 -*-

"""
ReferIt, UNC, UNC+ and GRef referring image segmentation PyTorch dataset.

Define and group batches of images, segmentations and queries.
Based on:
https://github.com/chenxi116/TF-phrasecut-public/blob/master/build_batches.py
"""

import os
import re
# import cv2
import sys
import json
from pathlib import Path
import torch
import numpy as np
import os.path as osp
import scipy.io as sio
import torch.utils.data as data
import pdb
import sng_parser

from utils.box_utils import xywh2xyxy
sys.path.append('.')

from PIL import Image, UnidentifiedImageError
#from pytorch_pretrained_bert.tokenization import BertTokenizer
#from transformers import BertTokenizer
from transformers import AutoTokenizer
from utils.word_utils import Corpus
import torch.nn.functional as F


SKYFIND_SPLIT_FILES = {
    'train': 'Train.json',
    'val': 'Val.json',
    'test': 'Test.json',
}

def read_examples(input_line, unique_id):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    # unique_id = 0
    line = input_line #reader.readline()
    # if not line:
    #     break
    line = line.strip()
    text_a = None
    text_b = None
    m = re.match(r"^(.*) \|\|\| (.*)$", line)
    if m is None:
        text_a = line
    else:
        text_a = m.group(1)
        text_b = m.group(2)
    examples.append(
        InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
    # unique_id += 1
    return examples

## Bert text encoding
class InputExample(object):
    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b
'''
class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids
'''
class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, unique_id, input_ids, input_mask):
        self.unique_id = unique_id
        #self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        #self.input_type_ids = input_type_ids

class InputFeatures_selection(object):
    """A single set of features of data."""
    def __init__(self, unique_id, input_ids, input_mask, words_mask):
        self.unique_id = unique_id
        #self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.words_mask = words_mask
        #self.input_type_ids = input_type_ids

def prompt_convert(examples, template = "This sentence : ' *sent 0* ' means<mask>."):
    # " This sentence : ' *sent 0* ' means<mask>." for sup-roberta
    # 'This sentence of "*sent 0*" means[MASK].' for sup-bert
    #prompt1 = 'This sentence of "'
    #prompt2 = '" means[MASK].'    
    #prompt_exapmple = prompt1 + examples + prompt2
    if len(examples) > 0 and examples[-1] not in '.?"\'': examples += '.'
    prompt_exapmple = template.replace('*sent 0*', examples).strip()

    return prompt_exapmple 


def _clamp_skyfind_bbox_xyxy(bbox_xyxy, image_width, image_height):
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    original_bbox = [x1, y1, x2, y2]
    x1 = min(max(x1, 0.0), float(image_width - 1))
    y1 = min(max(y1, 0.0), float(image_height - 1))
    x2 = min(max(x2, 0.0), float(image_width - 1))
    y2 = min(max(y2, 0.0), float(image_height - 1))
    clamped_bbox = [x1, y1, x2, y2]
    was_clamped = clamped_bbox != original_bbox
    is_valid = (x2 > x1) and (y2 > y1)
    return original_bbox, clamped_bbox, was_clamped, is_valid


def ralation_analysis(examples):
    graph = sng_parser.parse(examples)
    entities = graph['entities']
    relations = graph['relations']
    if len(entities) == 0:
        if len(relations) == 0:
            example_token = examples.split(' ')
            word_mask = np.ones(5 + len(example_token) + 6)
        else:
            
            selection = []
            example_token = examples.split(' ')
            word_mask_relation = np.zeros(len(example_token))
            for i in range(0, len(relations)):
                selection = selection + relations[i]['relation'].split(' ')

            for i in range(0, len(selection)):
                for j in range(0, len(example_token)):
                    if selection[i] == example_token[j]:
                        word_mask_relation[j] = 1

            word_mask_left = np.concatenate((np.ones(5), word_mask_relation))
            word_mask = np.concatenate((word_mask_left, np.ones(6)))
            
    else:
        if len(relations) == 0:
            #pdb.set_trace()
            selection = []
            example_token = examples.split(' ')
            word_mask_entity = np.zeros(len(example_token))
            for i in range(0, len(entities)):
                selection = selection + entities[i]['head'].split(' ')  

            for i in range(0, len(selection)):
                for j in range(0, len(example_token)):
                    if selection[i] == example_token[j]:
                        word_mask_entity[j] = 1
            
            word_mask_left = np.concatenate((np.ones(5), word_mask_entity))
            word_mask = np.concatenate((word_mask_left, np.ones(6)))
        
        else:
            selection = []
            example_token = examples.split(' ')
            word_mask_entity_relation = np.zeros(len(example_token))
            for i in range(0, len(entities)):
                selection = selection + entities[i]['head'].split(' ')   
            
            for i in range(0, len(relations)):
                selection = selection + relations[i]['relation'].split(' ')
           
            for i in range(0, len(selection)):
                for j in range(0, len(example_token)):
                    if selection[i] == example_token[j]:
                        word_mask_entity_relation[j] = 1

            word_mask_left = np.concatenate((np.ones(5), word_mask_entity_relation))
            word_mask = np.concatenate((word_mask_left, np.ones(6)))


    return list(map(int, word_mask.tolist()))


def _build_word_selection_mask(expression, batch_a, seq_length):
    attention_mask = list(np.array(batch_a['attention_mask'][0], dtype=np.int64))
    actual_text_len = int(sum(attention_mask))
    try:
        words_mask = ralation_analysis(expression)
        parser_fallback = False
    except Exception:
        words_mask = [1] * actual_text_len
        parser_fallback = True

    if len(words_mask) > actual_text_len:
        words_mask = words_mask[:actual_text_len]
    elif len(words_mask) < actual_text_len:
        words_mask = words_mask + [1] * (actual_text_len - len(words_mask))

    if len(words_mask) > seq_length:
        words_mask = words_mask[:seq_length]
    elif len(words_mask) < seq_length:
        words_mask = words_mask + [0] * (seq_length - len(words_mask))

    return list(map(int, words_mask)), parser_fallback, actual_text_len



# def convert_examples_to_features(examples, similar_examples, seq_length, tokenizer):
#     """Loads a data file into a list of `InputBatch`s."""
#     features = []
#     features_s = []
#     #features_info = []
#     for (ex_index, example) in enumerate(examples):
#         #list_sentence = []
        
#         prompt_tokens_a = prompt_convert(example.text_a)
#         prompt_tokens_a_s = prompt_convert(similar_examples)
#         words_mask = ralation_analysis(example.text_a)
#         #print('@@@@@@@@@@@@@@@@@@2')
#         #print(prompt_tokens_a)
#         #print(prompt_tokens_a_s)
#         #print('@@@@@@@@@@@@@@@@@@2')
#         #tokens_a = tokenizer.tokenize(prompt_tokens_a)
#         #tokens_a_s = tokenizer.tokenize(prompt_tokens_a_s)
#         #print('length', len(tokens_a), tokens_a)
#         #tokens_a = tokenizer.tokenize(example.text_a)
#         #tokens_c = tokenizer.whitespace_tokenize(example.text_a)

#         #print('sentence:', example.text_a)
#         #print('sentence:', example.text_a)
#         #print('type:', tokens_a[0])
#         #graph = sng_parser.parse(example.text_a)
#         #relations = graph['relations']
#         #entities = graph['entities']
#         #print(entities)
#         #batch_a = tokenizer.batch_encode_plus(prompt_tokens_a, return_tensors='pt', padding=True)



#         '''
#         tokens_b = None
#         if example.text_b:
#             tokens_b = tokenizer.tokenize(example.text_b)

#         if tokens_b:
#             # Modifies `tokens_a` and `tokens_b` in place so that the total
#             # length is less than the specified length.
#             # Account for [CLS], [SEP], [SEP] with "- 3"
#             _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
#         else:
#             # Account for [CLS] and [SEP] with "- 2"
#             if len(prompt_tokens_a) > seq_length - 2:
#                 prompt_tokens_a = prompt_tokens_a[0:(seq_length - 2)]
            
#             if len(prompt_tokens_a_s) > seq_length - 2:
#                 prompt_tokens_a_s = prompt_tokens_a_s[0:(seq_length - 2)]
            
        
        
#         tokens = []
#         input_type_ids = []
#         tokens.append("[CLS]")
#         input_type_ids.append(0)
#         for token in tokens_a:
#             tokens.append(token)
#             input_type_ids.append(0)
#         tokens.append("[SEP]")
#         input_type_ids.append(0)

#         if tokens_b:
#             for token in tokens_b:
#                 tokens.append(token)
#                 input_type_ids.append(1)
#             tokens.append("[SEP]")
#             input_type_ids.append(1)
        
#         tokens_a = []
#         tokens_a_s = []
#         tokens_a.append("[CLS]")
#         for token in prompt_tokens_a:
#             tokens_a.append(token)
#         tokens_a.append("[SEP]")

#         tokens_a_s.append("[CLS]")
#         for token in prompt_tokens_a_s:
#             tokens_a_s.append(token)
#         tokens_a_s.append("[SEP]")
#         '''

#         batch_a = tokenizer.batch_encode_plus([prompt_tokens_a], return_tensors='pt', padding=True)
#         batch_a_s = tokenizer.batch_encode_plus([prompt_tokens_a_s], return_tensors='pt', padding=True)
        
#         #print(batch_a)
#         #input_ids = tokenizer.convert_tokens_to_ids(tokens)
#         #sentence
#         input_ids = list(np.array(batch_a['input_ids'][0]))
#         input_mask = list(np.array(batch_a['attention_mask'][0]))
#         #similar sentence
#         input_ids_s = list(np.array(batch_a_s['input_ids'][0]))
#         input_mask_s = list(np.array(batch_a_s['attention_mask'][0]))
#         #input_type_ids = list(np.array(batch_a['token_type_ids'][0]))
#         #print('##################')
#         #print(input_ids)
#         #print(input_ids_s)
#         #print(input_ids)
#         #print('##################')
#         # The mask has 1 for real tokens and 0 for padding tokens. Only real
#         # tokens are attended to.
        

#         #input_mask = [1] * len(input_ids)

#         # Zero-pad up to the sequence length.
#         #sentence
#         while len(input_ids) < seq_length:
#             input_ids.append(0)
#             input_mask.append(0)
#             #input_type_ids.append(0)
#         while len(words_mask) < seq_length:
#             words_mask.append(0)


#         assert len(input_ids) == seq_length
#         assert len(input_mask) == seq_length
#         assert len(words_mask) == seq_length
#         #assert len(input_type_ids) == seq_length
#         features.append(
#             InputFeatures_selection(
#                 unique_id=example.unique_id,
#                 #tokens=tokens,
#                 input_ids=input_ids,
#                 input_mask=input_mask,
#                 words_mask=words_mask))
#                 #input_type_ids=input_type_ids))


#         #similar sentence
#         while len(input_ids_s) < seq_length:
#             input_ids_s.append(0)
#             input_mask_s.append(0)
#             #input_type_ids.append(0)

#         assert len(input_ids_s) == seq_length
#         assert len(input_mask_s) == seq_length
#         #assert len(input_type_ids) == seq_length
#         features_s.append(
#             InputFeatures(
#                 unique_id=example.unique_id,
#                 #tokens=tokens,
#                 input_ids=input_ids_s,
#                 input_mask=input_mask_s))


        
#     return features, features_s, batch_a, batch_a_s
    #return batch_a

def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    #features_info = []
    parser_fallback_count = 0
    token_truncated_count = 0
    for (ex_index, example) in enumerate(examples):
        #list_sentence = []
        
        prompt_tokens_a = prompt_convert(example.text_a)
        #print('@@@@@@@@@@@@@@@@@@2')
        #print(prompt_tokens_a)
        #print(prompt_tokens_a_s)
        #print('@@@@@@@@@@@@@@@@@@2')
        #tokens_a = tokenizer.tokenize(prompt_tokens_a)
        #tokens_a_s = tokenizer.tokenize(prompt_tokens_a_s)
        #print('length', len(tokens_a), tokens_a)
        #tokens_a = tokenizer.tokenize(example.text_a)
        #tokens_c = tokenizer.whitespace_tokenize(example.text_a)

        #print('sentence:', example.text_a)
        #print('sentence:', example.text_a)
        #print('type:', tokens_a[0])
        #graph = sng_parser.parse(example.text_a)
        #relations = graph['relations']
        #entities = graph['entities']
        #print(entities)
        #batch_a = tokenizer.batch_encode_plus(prompt_tokens_a, return_tensors='pt', padding=True)



        '''
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(prompt_tokens_a) > seq_length - 2:
                prompt_tokens_a = prompt_tokens_a[0:(seq_length - 2)]
            
            if len(prompt_tokens_a_s) > seq_length - 2:
                prompt_tokens_a_s = prompt_tokens_a_s[0:(seq_length - 2)]
            
        
        
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)
        
        tokens_a = []
        tokens_a_s = []
        tokens_a.append("[CLS]")
        for token in prompt_tokens_a:
            tokens_a.append(token)
        tokens_a.append("[SEP]")

        tokens_a_s.append("[CLS]")
        for token in prompt_tokens_a_s:
            tokens_a_s.append(token)
        tokens_a_s.append("[SEP]")
        '''

        batch_a = tokenizer.batch_encode_plus(
            [prompt_tokens_a],
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=seq_length,
        )
        #batch_a_s = tokenizer.batch_encode_plus([prompt_tokens_a_s], return_tensors='pt', padding=True)
        
        #print(batch_a)
        #input_ids = tokenizer.convert_tokens_to_ids(tokens)
        #sentence
        input_ids = list(np.array(batch_a['input_ids'][0]))
        input_mask = list(np.array(batch_a['attention_mask'][0]))
        if len(tokenizer.encode(prompt_tokens_a, add_special_tokens=True)) > seq_length:
            token_truncated_count += 1
        words_mask, parser_fallback, _ = _build_word_selection_mask(
            expression=example.text_a,
            batch_a=batch_a,
            seq_length=seq_length,
        )
        if parser_fallback:
            parser_fallback_count += 1

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        

        #input_mask = [1] * len(input_ids)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(words_mask) == seq_length
        #assert len(input_type_ids) == seq_length
        features.append(
            InputFeatures_selection(
                unique_id=example.unique_id,
                #tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                words_mask=words_mask))
                #input_type_ids=input_type_ids))


    
    batch_a['parser_fallback_count'] = parser_fallback_count
    batch_a['token_truncated_count'] = token_truncated_count
    return features, batch_a














class DatasetNotFoundError(Exception):
    pass

class TransVGDataset(data.Dataset):
    SUPPORTED_DATASETS = {
        'referit': {'splits': ('train', 'val', 'trainval', 'test')},
        'unc': {
            'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
            'params': {'dataset': 'refcoco', 'split_by': 'unc'}
        },
        'unc+': {
            'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
            'params': {'dataset': 'refcoco+', 'split_by': 'unc'}
        },
        'gref': {
            'splits': ('train', 'val'),
            'params': {'dataset': 'refcocog', 'split_by': 'google'}
        },
        'gref_umd': {
            'splits': ('train', 'val', 'test'),
            'params': {'dataset': 'refcocog', 'split_by': 'umd'}
        },
        'flickr': {
            'splits': ('train', 'val', 'test')},
        'skyfind': {
            'splits': ('train', 'val', 'test')
        },
    }

    def __init__(self, data_root, split_root='data', dataset='referit', 
                 transform=None, return_idx=False, testmode=False,
                 split='train', max_query_len=128, lstm=False, 
                 bert_model='auto',swin=False):
        self.images = []
        self.data_root = data_root
        self.split_root = split_root
        self.dataset = dataset
        self.query_len = max_query_len
        self.lstm = lstm
        self.transform = transform 
        self.testmode = testmode
        self.split = split
        self.skyfind_meta_by_img_id = {}
        self.skyfind_stats = {}
        #sup-bert
        #pdb.set_trace()
        #print('###############')
        #print(bert_model)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model, use_fast=True)
        #print('###############')
        #self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
        self.return_idx=return_idx
        self.swin=swin
        assert self.transform is not None

        if split == 'train':
            self.augment = True
        else:
            self.augment = False

        if self.dataset == 'referit':
            self.dataset_root = osp.join(self.data_root, 'referit')
            self.im_dir = osp.join(self.dataset_root, 'images')
            self.split_dir = osp.join(self.dataset_root, 'splits')
        elif  self.dataset == 'flickr':
            self.dataset_root = osp.join(self.data_root, 'Flickr30k')
            self.im_dir = osp.join(self.dataset_root, 'flickr30k_images')
        elif self.dataset == 'skyfind':
            self.dataset_root = self.data_root
            self.im_dir = osp.join(self.dataset_root, 'images')
        else:   ## refcoco, etc.
            self.dataset_root = osp.join(self.data_root, 'other')
            self.im_dir = osp.join(
                self.dataset_root, 'images', 'mscoco', 'images', 'train2014')
            self.split_dir = osp.join(self.dataset_root, 'splits')

        valid_splits = self.SUPPORTED_DATASETS[self.dataset]['splits']
        if split not in valid_splits:
            raise ValueError(
                'Dataset {0} does not have split {1}'.format(
                    self.dataset, split))

        if self.dataset == 'skyfind':
            self._init_skyfind_dataset()
            return

        if not self.exists_dataset():
            # self.process_dataset()
            print('Please download index cache to data folder: \n \
                https://drive.google.com/open?id=1cZI562MABLtAzM6YU4WmKPFFguuVr0lZ')
            exit(0)
        self.extended_data_path = {}
        dataset_path = osp.join(self.split_root, self.dataset)

        for i in range(0, len(valid_splits)):
            if valid_splits[i] != 'trainval':
                self.extended_data_path[valid_splits[i]] = dataset_path +'/' + valid_splits[i] + '_' + self.dataset + '.json'
        
        '''
        with open(self.extended_data_path[self.split], 'r', encoding='utf8') as f:
            #pdb.set_trace()
            self.json_data = json.load(f)
        '''
        
        #pdb.set_trace()

        if self.lstm:
            self.corpus = Corpus()
            corpus_path = osp.join(dataset_path, 'corpus.pth')
            self.corpus = torch.load(corpus_path)

        splits = [split]

        if self.dataset != 'referit':
            splits = ['train', 'val'] if split == 'trainval' else [split]
        for split in splits:
            imgset_file = '{0}_{1}.pth'.format(self.dataset, split)
            imgset_path = osp.join(dataset_path, imgset_file)
            #pdb.set_trace()
            self.images += torch.load(imgset_path)
            
            '''
            if split == 'train' or split == 'val' or split == 'testA' or split == 'testB':
                pdb.set_trace()
                self.images += torch.load(imgset_path)
                data = []
                for i in range(0, len(self.images)):
                    data.append([self.images[i][0], self.images[i][3]])
                pdb.set_trace()
                #np.save('./refcoco_train_info.npy', data)
            '''
            #pdb.set_trace()

    def exists_dataset(self):
        if self.dataset == 'skyfind':
            annotation_file = osp.join(self.dataset_root, SKYFIND_SPLIT_FILES[self.split])
            return osp.isdir(self.im_dir) and osp.isfile(annotation_file)
        return osp.exists(osp.join(self.split_root, self.dataset))

    def _init_skyfind_dataset(self):
        annotation_file = osp.join(self.dataset_root, SKYFIND_SPLIT_FILES[self.split])
        if not osp.isdir(self.im_dir):
            raise FileNotFoundError('SkyFind image directory not found: {}'.format(self.im_dir))
        if not osp.isfile(annotation_file):
            raise FileNotFoundError('SkyFind annotation file not found: {}'.format(annotation_file))

        with open(annotation_file, 'r', encoding='utf8') as f:
            raw_samples = json.load(f)

        if not isinstance(raw_samples, list):
            raise ValueError('SkyFind annotation file must contain a list: {}'.format(annotation_file))

        unique_file_names = sorted({sample['fileName'] for sample in raw_samples})
        image_info_by_file = {}
        missing_files = set()
        corrupt_files = set()

        for file_name in unique_file_names:
            image_path = osp.join(self.im_dir, file_name)
            if not osp.isfile(image_path):
                missing_files.add(file_name)
                continue
            try:
                with Image.open(image_path) as image:
                    image_width, image_height = image.size
            except (UnidentifiedImageError, OSError, ValueError):
                corrupt_files.add(file_name)
                continue
            image_info_by_file[file_name] = {
                'image_path': image_path,
                'image_width': image_width,
                'image_height': image_height,
            }

        skipped_missing = 0
        skipped_corrupt = 0
        skipped_invalid_bbox = 0
        clamped_bbox_count = 0
        non_numeric_file_name_count = 0

        for sample in raw_samples:
            file_name = sample['fileName']
            if file_name in missing_files:
                skipped_missing += 1
                continue
            if file_name in corrupt_files:
                skipped_corrupt += 1
                continue

            image_info = image_info_by_file[file_name]
            image_width = image_info['image_width']
            image_height = image_info['image_height']
            expression = sample['expression']
            if not isinstance(expression, str) or not expression.strip():
                skipped_invalid_bbox += 1
                continue

            original_bbox, clamped_bbox, was_clamped, is_valid = _clamp_skyfind_bbox_xyxy(
                sample['bbox'],
                image_width=image_width,
                image_height=image_height,
            )
            if not is_valid:
                skipped_invalid_bbox += 1
                continue
            if was_clamped:
                clamped_bbox_count += 1
            if not Path(file_name).stem.isdigit():
                non_numeric_file_name_count += 1

            img_id = len(self.images)
            raw_img_id = Path(file_name).stem
            sample_record = {
                'img_id': img_id,
                'file_name': file_name,
                'raw_img_id': raw_img_id,
                'image_path': image_info['image_path'],
                'image_width': image_width,
                'image_height': image_height,
                'expression': expression.strip(),
                'bbox_xyxy_raw_original': original_bbox,
                'bbox_xyxy_raw_clamped': clamped_bbox,
                'bbox_was_clamped': was_clamped,
            }
            self.images.append(sample_record)
            self.skyfind_meta_by_img_id[img_id] = {
                'file_name': file_name,
                'raw_img_id': raw_img_id,
                'expression': expression.strip(),
                'gt_bbox_xyxy_raw_original': original_bbox,
                'gt_bbox_xyxy_raw_clamped': clamped_bbox,
                'bbox_was_clamped': was_clamped,
                'orig_size': (image_height, image_width),
            }

        self.skyfind_stats = {
            'split': self.split,
            'annotation_file': annotation_file,
            'raw_samples': len(raw_samples),
            'kept_samples': len(self.images),
            'unique_images': len(unique_file_names),
            'validated_images': len(image_info_by_file),
            'missing_images': len(missing_files),
            'corrupt_images': len(corrupt_files),
            'skipped_missing_samples': skipped_missing,
            'skipped_corrupt_samples': skipped_corrupt,
            'skipped_invalid_bbox_samples': skipped_invalid_bbox,
            'clamped_bbox_samples': clamped_bbox_count,
            'non_numeric_file_name_samples': non_numeric_file_name_count,
        }

        print(
            '[SkyFindDataset] split={split} raw_samples={raw_samples} kept_samples={kept_samples} '
            'unique_images={unique_images} validated_images={validated_images} '
            'missing_images={missing_images} corrupt_images={corrupt_images} '
            'skipped_missing_samples={skipped_missing_samples} '
            'skipped_corrupt_samples={skipped_corrupt_samples} '
            'skipped_invalid_bbox_samples={skipped_invalid_bbox_samples} '
            'clamped_bbox_samples={clamped_bbox_samples} '
            'non_numeric_file_name_samples={non_numeric_file_name_samples}'.format(**self.skyfind_stats)
        )

    def pull_skyfind_item(self, idx):
        sample = self.images[idx]
        img = Image.open(sample['image_path']).convert("RGB")
        if self.swin:
            img = Image.fromarray(np.asarray(img)[..., ::-1])
        bbox = torch.tensor(sample['bbox_xyxy_raw_clamped']).float()
        return sample['img_id'], img, sample['expression'], bbox

    def pull_item(self, idx):
        # x1,y1,x2,y2
        if self.dataset == 'skyfind':
            return self.pull_skyfind_item(idx)
        if self.dataset == 'flickr':
            img_file, bbox, phrase = self.images[idx]
        else:
            
            img_file, _, bbox, phrase, attri = self.images[idx]

        img_id =  int(img_file.split('_')[2].split('.')[0])
        #print(img_id)
        ## box format: to x1y1x2y2
        if not (self.dataset == 'referit' or self.dataset == 'flickr'):
            bbox = np.array(bbox, dtype=int)
            bbox[2], bbox[3] = bbox[0]+bbox[2], bbox[1]+bbox[3]
        else:
            bbox = np.array(bbox, dtype=int)

        img_path = osp.join(self.im_dir, img_file)
        img = Image.open(img_path).convert("RGB")
        if self.swin:
            # Swin-Transformer need BGR image
            img=Image.fromarray(np.asarray(img)[...,::-1])


        bbox = torch.tensor(bbox)
        bbox = bbox.float()
        return img_id, img, phrase, bbox 

    def tokenize_phrase(self, phrase):
        return self.corpus.tokenize(phrase, self.query_len)

    def untokenize_word_vector(self, words):
        return self.corpus.dictionary[words]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        #print('############################################')
        #print(idx)
        #print('############################################')
        from icecream import ic
        img_id, img, phrase, bbox = self.pull_item(idx)
        #similar_sentence = self.json_data[str(idx)][1]


        # print('############################################')
        # print("image", img)
        # print("phrase", phrase)
        # print("bbox", bbox)
        # print('############################################')
        # phrase = phrase.decode("utf-8").encode().lower()
        phrase = phrase.lower()
        input_dict = {'img': img, 'box': bbox, 'text': phrase, 'img_id':img_id}
        
        input_dict = self.transform(input_dict)
        img = input_dict['img']
        img_id = input_dict['img_id']
        h, w = img.shape[-2:]
        #print(h,w)
        bbox = input_dict['box']
        phrase = input_dict['text']
        img_mask = input_dict['mask'] 

        if self.lstm:
            phrase = self.tokenize_phrase(phrase)
            word_id = phrase
            word_mask = np.array(word_id>0, dtype=int)
        else:
            ## encode phrase to bert input
            examples = read_examples(phrase, idx)
            #print("###################")
            #print(phrase, idx, img_id)
            #print("###################")
            # features, features_s, batch_a, batch_a_s = convert_examples_to_features(
            #     examples=examples, similar_examples=similar_sentence, seq_length=self.query_len, tokenizer=self.tokenizer)
            
            features, batch_a = convert_examples_to_features(
                examples=examples,seq_length=self.query_len, tokenizer=self.tokenizer)
            word_id = features[0].input_ids
            word_mask = features[0].input_mask
            word_selection = features[0].words_mask
            #word_id_s = features_s[0].input_ids
            #word_mask_s = features_s[0].input_mask
           

            #input_type_ids = features[0].input_type_ids
            #word_id = features['input_ids']
            #word_mask = features['attention_mask']
        
        if self.testmode:
            if self.dataset == 'skyfind':
                file_name = self.images[idx]['file_name']
                raise NotImplementedError(
                    'SkyFind testmode export path is not implemented yet for {}. '
                    'Use normal evaluation mode or add an explicit export script.'.format(file_name)
                )
            return img, img_id, np.array(word_id, dtype=int), np.array(word_mask, dtype=int), \
                np.array(bbox, dtype=np.float32), np.array(ratio, dtype=np.float32), \
                np.array(dw, dtype=np.float32), np.array(dh, dtype=np.float32), self.images[idx][0]
        else:
            if self.split=='train':
                #print('############################################')
                #print("phrase", np.array(word_id, dtype=int))
                #print("bbox", np.array(word_mask, dtype=int))
                #print('############################################')
                
                #return img, img_id, np.array(img_mask), np.array(word_id, dtype=int), np.array(word_selection, dtype=int), np.array(word_id_s, dtype=int), batch_a, batch_a_s, np.array(word_mask, dtype=int), np.array(word_mask_s, dtype=int), np.array(bbox, dtype=np.float32)
                return img, img_id, np.array(img_mask), np.array(word_id, dtype=int), np.array(word_selection, dtype=int), batch_a, np.array(word_mask, dtype=int), np.array(bbox, dtype=np.float32)
            else:
                #return img, img_id, np.array(img_mask), np.array(word_id, dtype=int), np.array(word_selection, dtype=int), np.array(word_id_s, dtype=int), batch_a, batch_a_s, np.array(word_mask, dtype=int), np.array(word_mask_s, dtype=int), np.array(bbox, dtype=np.float32)
                return img, img_id, np.array(img_mask), np.array(word_id, dtype=int), np.array(word_selection, dtype=int), batch_a, np.array(word_mask, dtype=int), np.array(bbox, dtype=np.float32)
