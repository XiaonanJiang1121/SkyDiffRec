'''
import torch
import numpy as np
import pdb
path = 'data/unc/train_unc.json'
data = []
image = torch.load(path)
pdb.set_trace()
for i in range(0, len(image)):
    print(i)
    data.append([image[i][0], image[i][3]])
                
np.save('./refcoco_test_info.npy', data)
#print(len(image))
'''
'''
import torch
state_dict = torch.load("resnet18.pth")
print(type(state_dict))
'''

import sng_parser
from pprint import pprint
import numpy as np
import pdb
import json


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


    return word_mask


#example = 'in in on'
#word_mask = ralation_analysis(example)

with open('train_unc.json', 'r', encoding='utf8') as f:
    json_data = json.load(f)
    keys = list(json_data.keys())
    for i in range(0, len(keys)):
        print(i)
        i_expression = json_data[keys[i]][0]
        print(i_expression)
        word_mask = ralation_analysis(i_expression)
        pdb.set_trace()
        print(list(map(int, word_mask.tolist())))






