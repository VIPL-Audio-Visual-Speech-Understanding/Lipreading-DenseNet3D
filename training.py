from torch.autograd import Variable
import torch
import torch.optim as optim
from datetime import datetime, timedelta
from data import LipreadingDataset
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torch.nn as nn
import os
import pdb
import math


def timedelta_string(timedelta):
    totalSeconds = int(timedelta.total_seconds())
    hours, remainder = divmod(totalSeconds,60*60)
    minutes, seconds = divmod(remainder,60)
    return "{:0>2} hrs, {:0>2} mins, {:0>2} secs".format(hours, minutes, seconds)

def output_iteration(loss, i, time, totalitems):

    avgBatchTime = time / (i+1)
    estTime = avgBatchTime * (totalitems - i)
    
    print("Iteration: {:0>8},Elapsed Time: {},Estimated Time Remaining: {},Loss:{}".format(i, timedelta_string(time), timedelta_string(estTime),loss))

class Trainer():

    tot_iter = 0
    writer = SummaryWriter()    
    
    def __init__(self, options):
                                
        self.usecudnn = options["general"]["usecudnn"]

        self.batchsize = options["input"]["batchsize"]

        self.statsfrequency = options["training"]["statsfrequency"]       

        self.learningrate = options["training"]["learningrate"]

        self.modelType = options["training"]["learningrate"]

        self.weightdecay = options["training"]["weightdecay"]
        self.momentum = options["training"]["momentum"]
        
        self.save_prefix = options["training"]["save_prefix"]
        
        self.trainingdataset = LipreadingDataset(options["training"]["data_root"], 
                                                options["training"]["index_root"], 
                                                options["training"]["padding"], 
                                                True)
        
        self.trainingdataloader = DataLoader(
                                    self.trainingdataset,
                                    batch_size=options["input"]["batchsize"],
                                    shuffle=options["input"]["shuffle"],
                                    num_workers=options["input"]["numworkers"],
                                    drop_last=True)
        

    def learningRate(self, epoch):
        decay = math.floor((epoch - 1) / 5)
        return self.learningrate * pow(0.5, decay)

    def __call__(self, model, epoch):
        #set up the loss function.
        model.train()
        criterion = model.loss()
        if(self.usecudnn):
            net = nn.DataParallel(model).cuda()
        criterion = criterion.cuda()
               
        optimizer = optim.Adam(
                        model.parameters(),
                        lr = self.learningrate, amsgrad=True)
        
        #transfer the model to the GPU.       
            
        startTime = datetime.now()
        print("Starting training...")
        for i_batch, sample_batched in enumerate(self.trainingdataloader):
            optimizer.zero_grad()
            input = Variable(sample_batched['temporalvolume'])
            labels = Variable(sample_batched['label'])
            length = Variable(sample_batched['length'])
            
            if(self.usecudnn):
                input = input.cuda()
                labels = labels.cuda()

            outputs = net(input)
            loss = criterion(outputs, length, labels.squeeze(1))
            loss.backward()
            optimizer.step()
            sampleNumber = i_batch * self.batchsize

            if(sampleNumber % self.statsfrequency == 0):
                currentTime = datetime.now()
                output_iteration(loss.cpu().detach().numpy(), sampleNumber, currentTime - startTime, len(self.trainingdataset))
                Trainer.writer.add_scalar('Train Loss', loss, Trainer.tot_iter)
            Trainer.tot_iter += 1

        print("Epoch completed, saving state...")
        torch.save(model.state_dict(), "{}_{:0>8}.pt".format(self.save_prefix, epoch))       
