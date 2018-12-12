from torch.autograd import Variable
import torch
import torch.optim as optim
from datetime import datetime, timedelta
from data import LipreadingDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import os
import numpy as np


class Validator():
    def __init__(self, options, mode):
    
        self.usecudnn = options["general"]["usecudnn"]

        self.batchsize = options["input"]["batchsize"]        
        self.validationdataset = LipreadingDataset(options[mode]["data_root"], 
                                                options[mode]["index_root"], 
                                                options[mode]["padding"],                                                
                                                False)
                                                
        self.tot_data = len(self.validationdataset)
        self.validationdataloader = DataLoader(
                                    self.validationdataset,
                                    batch_size=options["input"]["batchsize"],
                                    shuffle=options["input"]["shuffle"],
                                    num_workers=options["input"]["numworkers"],
                                    drop_last=False
                                )
        self.mode = mode
        
    def __call__(self, model):
        with torch.no_grad():
            print("Starting {}...".format(self.mode))
            count = np.zeros((len(self.validationdataset.pinyins)))
            validator_function = model.validator_function()
            model.eval()
            if(self.usecudnn):
                net = nn.DataParallel(model).cuda()
                
            num_samples = 0            
            for i_batch, sample_batched in enumerate(self.validationdataloader):
                input = Variable(sample_batched['temporalvolume']).cuda()
                labels = Variable(sample_batched['label']).cuda()
                length = Variable(sample_batched['length']).cuda()
                
                model = model.cuda()

                outputs = net(input)
                (vector, top1) = validator_function(outputs, length, labels)
                _, maxindices = vector.cpu().max(1)
                argmax = (-vector.cpu().numpy()).argsort()
                for i in range(input.size(0)):
                    p = list(argmax[i]).index(labels[i])
                    count[p:] += 1                    
                num_samples += input.size(0)
                

                print('i_batch/tot_batch:{}/{},corret/tot:{}/{},current_acc:{}'.format(i_batch,len(self.validationdataloader),count[0],len(self.validationdataset),1.0*count[0]/num_samples))                

        return count/num_samples
