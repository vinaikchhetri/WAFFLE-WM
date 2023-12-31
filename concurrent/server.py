from torch.utils.data import Dataset
import torch
import models
import torch.optim as optim
from torchvision.models import resnet18
from torchvision.models import vgg16
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
        self.pretrainingR = self.args.pretrainingR
        self.watermarking_loss_before_embedding = []
        self.watermarking_acc_before_embedding = []
        self.watermarking_loss_after_embedding = []
        self.watermarking_acc_after_embedding = []
        self.num_samples_dict = {} #number of samples per user.
        self.clients = []
        self.adversary_indices = []
        self.count_adv = []

        self.trainset,self.testset,self.client_data_dict,self.watermark_set = data_splitter.splitter(self.args)
        self.test_data_loader = torch.utils.data.DataLoader(self.testset, batch_size=64, shuffle=True)
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
            if self.args.dataset == "cifar-10":
                self.model_global = models.ResNet(18, num_classes = 10)
            else:
                self.model_global = models.ResNet(18, num_classes = 100)
            self.criterion = torch.nn.CrossEntropyLoss()
            self.model_global.to(self.device)
    
        if self.model == 'MNIST_L5':
            self.model_global = models.MNIST_L5()
            self.criterion = torch.nn.CrossEntropyLoss()
            self.model_global.to(self.device)

        if self.model == 'vgg':
            self.model_global = vgg16(pretrained=True, progress=True)
            self.criterion = torch.nn.CrossEntropyLoss()
            self.model_global.to(self.device)
        
    def load_model(self, model_global):
        self.model_local.load_state_dict(model_global.state_dict())

    def create_clients(self):
        self.adversary_indices = np.random.choice(self.K, self.args.num_attackers, replace=False) # Randomly sample some adversaries from the clients.
        is_adversary = 0 # This is used determine if the client is adversary or not.
        for i in range (self.K): #loop through clients
            if i in self.adversary_indices:
                is_adversary = 1
            else:
                is_adversary = 0
            self.num_samples_dict[i] = len(self.client_data_dict[i]) # Number of samples a client has.
            self.clients.append(client.Client(self.client_data_dict[i], self.trainset, self.args, self.device, is_adversary)) # Construct the client.

    # Main loop.
    def train(self):
        self.pretrain() # pretrain over the watermarking dataset.
        initial = time.time()
        test_acc_before_embedding = []
        test_loss_before_embedding = []
        test_acc_after_embedding = []
        test_loss_after_embedding = []

        for rounds in range(self.T): #total number of rounds
            if self.C == 0: # If C = 0 we set number of clients participating in current round to 1.
                m = 1 
            else:
                m = int(max(self.C*self.K, 1)) 

            client_indices = np.random.choice(self.K, m, replace=False)
            client_indices.astype(int)
            overlap = np.intersect1d(client_indices, self.adversary_indices) # Find the overlap between the particiapting clients and the adversaries.
            self.count_adv.append(len(overlap)) # Save a history of the number of adversaries per round.
            num_samples_list = [self.num_samples_dict[idx] for idx in client_indices] # list containing number of samples for each particiapting user id.
            total_num_samples = reduce(lambda x,y: x+y, num_samples_list, 0) # Total number of samples in the current round. 
            store = {} # Dictionary to store weights of user models. Store is indexed using the user id.

            avg_loss= 0 
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(client_indices), os.cpu_count() - 1)) as executor:
                results = []
                for index,client_idx in enumerate(client_indices): #loop through selected clients
                    self.clients[client_idx].load_model(self.model_global) # Load client model using global model.
                    results.append(executor.submit(self.clients[client_idx].client_update).result())            
                # Retrieve results as they become available
                for ind,future in enumerate(results):
                    store[ind] = future.state_dict()
                    # store[ind] = future[0].state_dict()
                    #avg_loss+=future[1]
                #avg_loss/=len(results)

            # Aggregate.
            w_global = {}
            for layer in store[0]:
                sum = 0
                for user_key in store:
                    sum += store[user_key][layer]*num_samples_list[user_key]/total_num_samples
                w_global[layer] = sum
            
            self.model_global.load_state_dict(w_global) # Update weigthts of the global model.

            #----Evaluation----

            rl,ra = self.test() # Performing evaluation on main task test data before retraining.
            test_loss_before_embedding.append(rl) 
            test_acc_before_embedding.append(ra)
        
            print('Round '+ str(rounds))
            print()
            print("Number of adversaries participating in this round: ", len(overlap))
            print()
            print('Before watermarking, main task')
            print(f'server stats: [test-loss: {rl:.3f}')
            print(f'server stats: [test-accuracy: {ra:.3f}')
            print()

            # Retrain the global model using the watermarks
            if not self.retrainingR == 0:
                self.retrain()
            rl,ra = self.test() # Performing evaluation on main task test data after retraining.
            test_loss_after_embedding.append(rl)
            test_acc_after_embedding.append(ra)

            print('After watermarking, main task')
            print(f'server stats: [test-loss: {rl:.3f}')
            print(f'server stats: [test-accuracy: {ra:.3f}')
            print()

        print("finished ", time.time() - initial)
        torch.save(test_loss_before_embedding,'../stats/main/test_loss_before_embedding_'+self.args.name+'.pt')
        torch.save(test_acc_before_embedding,'../stats/main/test_acc_before_embedding_'+self.args.name+'.pt')
        torch.save(test_loss_after_embedding,'../stats/main/test_loss_after_embedding_'+self.args.name+'.pt')
        torch.save(test_acc_after_embedding,'../stats/main/test_acc_after_embedding_'+self.args.name+'.pt')

        torch.save(self.watermarking_loss_before_embedding, '../stats/wm/watermarking_loss_before_embedding_'+self.args.name+'.pt')
        torch.save(self.watermarking_acc_before_embedding, '../stats/wm/watermarking_acc_before_embedding_'+self.args.name+'.pt')
        torch.save(self.watermarking_loss_after_embedding, '../stats/wm/watermarking_loss_after_embedding_'+self.args.name+'.pt')
        torch.save(self.watermarking_acc_after_embedding, '../stats/wm/watermarking_acc_after_embedding_'+self.args.name+'.pt')
        
        torch.save(self.model_global.state_dict(), '../stats/models/'+self.args.name+'.pt')
        torch.save(self.count_adv, '../stats/count/count_adv_'+self.args.name+'.pt')
        
    #----Performing evaluation on main task test data----
    def test(self):
        with torch.no_grad():
            self.model_global.eval()
            running_loss = 0.0
            running_acc = 0.0
            for index,data in enumerate(self.test_data_loader):  
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
                running_loss += loss.item()
        return running_loss/(index+1), running_acc/(index+1)

    #----Performing evaluation on watermark data----
    def watermark_test(self):
        with torch.no_grad():
            self.model_global.eval()
            running_loss = 0.0
            running_acc = 0.0
            for index,data in enumerate(self.watermark_loader):  
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
                running_loss += loss.item()
        return running_loss/(index+1), running_acc/(index+1)
            
    # retraining the global model after each aggregation round.        
    def retrain(self):
        tr = 0
        self.model_global.train()
        if self.args.dataset == "mnist":
            self.optimizer = optim.SGD(self.model_global.parameters(), lr=0.005)
        else:
            self.optimizer = optim.SGD(self.model_global.parameters(), lr=0.0005)
        
        # Performing evaluation on watermarking data before retraining.
        rl,ra = self.watermark_test()
        running_loss = rl
        running_acc = ra
        print("Watermarking Accuracy before watermarking")
        print(f'server stats: [watermarking-loss: {running_loss:.3f}')
        print(f'server stats: [watermarking-accuracy: {running_acc:.3f}')
        self.watermarking_loss_before_embedding.append(running_loss)
        self.watermarking_acc_before_embedding.append(running_acc)
        print()

        # Retraining loop.
        while(tr<self.retrainingR and running_acc < 98):
            tr+=1                        
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
                acc = utils.accuracy(pred, labels)
                running_acc += acc
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            running_acc = running_acc/(index+1)
            running_loss = running_loss/(index+1)
        
        # Performing evaluation on watermarking data after retraining.
        print('Watermarking Accuracy after watermarking.')
        print(f'server stats: [watermarking-loss: {running_loss:.3f}')
        print(f'server stats: [watermarking-accuracy: {running_acc:.3f}')
        self.watermarking_loss_after_embedding.append(running_loss)
        self.watermarking_acc_after_embedding.append(running_acc)
        print()

    # Pretrain over the watermarking data.
    def pretrain(self):
        tr = 0
        self.model_global.train()
        if self.args.dataset == "mnist":
            self.optimizer = optim.SGD(self.model_global.parameters(), lr=0.1, momentum=0.5, weight_decay=0.00005)
        else:
            self.optimizer = optim.SGD(self.model_global.parameters(), lr=0.0005, momentum=0.5, weight_decay=0.00005)

        while(tr<self.pretrainingR):
            tr+=1                        
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
                acc = utils.accuracy(pred, labels)
                running_acc += acc
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                loss.backward()
                self.optimizer.step()
        
        print('pre-training')
        print(f'server stats: [watermarking-loss: {running_loss/(index+1):.3f}')
        print(f'server stats: [watermarking-accuracy: {running_acc/(index+1):.3f}')
        self.watermarking_loss_before_embedding.append(running_loss/(index+1))
        self.watermarking_acc_before_embedding.append(running_acc/(index+1))
        print()
        