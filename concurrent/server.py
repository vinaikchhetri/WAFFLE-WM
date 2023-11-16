from torch.utils.data import Dataset
import torch
import models
import torch.optim as optim
from torchvision.models import resnet18
import data_splitter
import client
import utils

import numpy as np
import time
from functools import reduce
import concurrent.futures
import os

class Server():
    def __init__(self, args):
        self.args = args
        self.K = self.args.K
        self.T = self.args.T
        self.C = self.args.C
        self.retrainingR = self.args.retrainingR
        self.num_samples_dict = {} #number of samples per user.
        self.clients = []
        self.trainset,self.testset,self.client_data_dict,self.watermark_set = data_splitter.splitter(self.args)
        self.data_loader = torch.utils.data.DataLoader(self.testset, batch_size=64, shuffle=True)
        self.watermark_loader = torch.utils.data.DataLoader(self.watermark_set, batch_size=50, shuffle = True)
        if self.args.gpu == "gpu":
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        self.model = args.model
        if self.model == 'nn':
            self.model_global = models.MP(28*28,200,10)
            self.criterion = torch.nn.CrossEntropyLoss()
            self.model_global.to(self.device)
         
        if self.model == 'cnn':
            self.model_global = models.CNN_MNIST()
            self.criterion = torch.nn.CrossEntropyLoss()
            self.model_global.to(self.device)
        
        if self.model == 'resnet':
            self.model_global = resnet18(num_classes=100)
            self.criterion = torch.nn.CrossEntropyLoss()
            self.model_global.to(self.device)
        
    def load_model(self, model_global):
        self.model_local.load_state_dict(model_global.state_dict())

    def create_clients(self):
        for i in range (self.K): #loop through clients
            self.num_samples_dict[i] = len(self.client_data_dict[i]) 
            self.clients.append(client.Client(self.client_data_dict[i], self.trainset, self.args, self.device))


    def train(self):
        #best_test_acc = -1
        initial = time.time()
        for rounds in range(self.T): #total number of rounds
            if self.C == 0:
                m = 1 
            else:
                m = int(max(self.C*self.K, 1)) 

            client_indices = np.random.choice(self.K, m, replace=False)
            client_indices.astype(int)
            num_samples_list = [self.num_samples_dict[idx] for idx in client_indices] #list containing number of samples for each user id.
            total_num_samples = reduce(lambda x,y: x+y, num_samples_list, 0)
            store = {}
            avg_loss= 0 
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(client_indices), os.cpu_count() - 1)) as executor:
                results = []
                for index,client_idx in enumerate(client_indices): #loop through selected clients
                    self.clients[client_idx].load_model(self.model_global)
                    results.append(executor.submit(self.clients[client_idx].client_update).result())            
                # Retrieve results as they become available
                for ind,future in enumerate(results):
                    store[ind] = future[0].state_dict()
                    avg_loss+=future[1]
                avg_loss/=len(results)

            w_global = {}
            for layer in store[0]:
                sum = 0
                for user_key in store:
                    sum += store[user_key][layer]*num_samples_list[user_key]/total_num_samples
                w_global[layer] = sum
            
            self.model_global.load_state_dict(w_global)

            #Performing evaluation on test data.
            rl,ra = self.test()
            print('Before watermarking')
            print('Round '+ str(rounds))
            print(f'server stats: [test-loss: {rl:.3f}')
            print(f'server stats: [train-loss: { avg_loss:.3f}')
            print(f'server stats: [accuracy: {ra:.3f}')
            

            #Retrain the server on the watermarks
            if not self.retrainingR == 0:
                self.retrain()

            #Performing evaluation on test data.
            rl,ra = self.test()

            print('After watermarking')
            print('Round '+ str(rounds))
            print(f'server stats: [test-loss: {rl:.3f}')
            print(f'server stats: [train-loss: { avg_loss:.3f}')
            print(f'server stats: [accuracy: {ra:.3f}')

            print()
        print("finished ", time.time() - initial)


    def test(self):
        with torch.no_grad():
            self.model_global.eval()
            running_loss = 0.0
            running_acc = 0.0
            for index,data in enumerate(self.data_loader):  
                inputs, labels = data
                inputs = inputs.to(self.device)
                if self.model == 'nn':
                    inputs = inputs.flatten(1)
                labels = labels.to(self.device)
                output = self.model_global(inputs)
                loss = self.criterion(output, labels)
                pred = torch.argmax(output, dim=1)
                acc = utils.accuracy(pred, labels)
                running_acc += acc
                running_loss += loss
        return running_loss/index+1, running_acc/index+1
            
    #retraining the global model after each aggregation round.        
    def retrain(self):
        tr = 0
        while(tr<self.retrainingR):
            tr+=1
            self.model_global.train()
            self.optimizer = optim.SGD(self.model_global.parameters(), lr=0.01, momentum=0.5)
            
            for epoch in range(self.args.E):
                running_loss = 0.0
                running_acc = 0.0
                for index,data in enumerate(self.watermark_loader):
                    inputs, labels = data
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    self.optimizer.zero_grad()
                    if self.args.model == 'nn':
                        inputs = inputs.flatten(1)
                    outputs = self.model_global(inputs)
                    pred = torch.argmax(outputs, dim=1)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
        # return self.model_global, loss