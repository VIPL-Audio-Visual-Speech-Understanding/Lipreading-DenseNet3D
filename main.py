from __future__ import print_function
from models.Dense3D import Dense3D
import torch
import toml
from training import Trainer
from validation import Validator
import torch.nn as nn
import os
import sys
from collections import OrderedDict   
import csv
import numpy as np
import json
import scipy.io as sio
from collections import defaultdict
import matplotlib.pyplot as plt


print("Loading options...")
with open(sys.argv[1], 'r') as optionsFile:
    options = toml.loads(optionsFile.read())

if(options["general"]["usecudnnbenchmark"] and options["general"]["usecudnn"]):
    print("Running cudnn benchmark...")
    torch.backends.cudnn.benchmark = True

os.environ['CUDA_VISIBLE_DEVICES'] = options["general"]['gpuid']
    
torch.manual_seed(options["general"]['random_seed'])

#Create the model.
model = Dense3D(options)

if(options["general"]["loadpretrainedmodel"]):
    # remove paralle module
    pretrained_dict = torch.load(options["general"]["pretrainedmodelpath"])
    # load only exists weights
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}
    print('matched keys:',len(pretrained_dict))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

#Move the model to the GPU.
criterion = model.loss()        
if(options["general"]["usecudnn"]):        
    torch.cuda.manual_seed(options["general"]['random_seed'])
    torch.cuda.manual_seed_all(options["general"]['random_seed'])

if(options["training"]["train"]):
    trainer = Trainer(options)
if(options["validation"]["validate"]):   
    validator = Validator(options, 'validation')
if(options['test']['test']):   
    tester = Validator(options, 'test')
    
for epoch in range(options["training"]["startepoch"], options["training"]["epochs"]):
    if(options["training"]["train"]):
        trainer(model, epoch)
    if(options["validation"]["validate"]):        
        result = validator(model)
        print('-'*21)
        print('{:<10}|{:>10}'.format('Top #', 'Accuracy'))
        for i in range(5):
            print('{:<10}|{:>10}'.format(i, result[i]))
        print('-'*21)
            
    if(options['test']['test']):
        result = tester(model)
        print('-'*21)
        print('{:<10}|{:>10}'.format('Top #', 'Accuracy'))
        for i in range(5):
            print('{:<10}|{:>10}'.format(i, result[i]))
        print('-'*21)
    
Trainer.writer.close()