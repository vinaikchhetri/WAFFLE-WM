import sys
from models import MP,CNN_CIFAR
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import tqdm
import torch.optim as optim
from options import arg_parser
from functools import reduce
from torch.utils.data import Dataset
import utils
import models
from threading import Thread

store = {}

class CustomDataset(Dataset):
	def __init__(self, dataset, idxs):
		self.dataset = dataset
		self.idxs = idxs
	def __len__(self):
		return len(self.idxs)
	def __getitem__(self, idx):
		tup = self.dataset[self.idxs[idx]]
		img = tup[0]
		label = tup[1]
		#return torch.tensor(img), torch.tensor(label)
		return img, label

def client_update(trainset, order_idx, client_idx, w_global, args, client_data_dict, criterion, model_local):
   model_local.load_state_dict(w_global)
   data_client = client_data_dict[client_idx] #client i's data.
   cd = CustomDataset(trainset, data_client)
   if args.B == 8:
      bs = len(trainset)
   else:
      bs = args.B
   data_loader = torch.utils.data.DataLoader(cd, batch_size=bs,
                                                shuffle=True)
   if args.gpu == "gpu":
      device = torch.device('cuda:0')
   else:
      device = torch.device('cpu')
	
   model_local.to(device)
   optimizer = optim.SGD(model_local.parameters(), lr=0.01, momentum=0.5)

   for epoch in range(args.E):
      running_loss = 0.0
      running_acc = 0.0
      for index,data in enumerate(data_loader):        
         inputs, labels = data
         #print("inputs ", inputs.shape)
         #print("labels ", labels)
         inputs = inputs.to(device)
         labels = labels.to(device)

      
         optimizer.zero_grad()
         if args.model == 'nn':
            inputs = inputs.flatten(1)
         outputs = model_local(inputs)
         pred = torch.argmax(outputs, dim=1)
         loss = criterion(outputs, labels)
         loss.backward()
         optimizer.step()

         # print statistics
         acc = utils.accuracy(pred, labels)
         running_acc += acc
         running_loss += loss.item()
         if index % 100 == 99:    # print every 2000 mini-batches
            #print(f'device: {client_idx} [{epoch + 1}, {index + 1:5d}] loss: {running_loss / 100:.3f}')
            #print(f'device: {client_idx} [{epoch + 1}, {index + 1:5d}] accuracy: {running_acc / 100:.3f}')
            running_loss = 0.0
            running_acc = 0.0
   # print('Finished Training Device '+ str(client_idx))
   store[order_idx] = model_local.state_dict()


if __name__=='__main__':
   args = arg_parser()
   if args.algo == "FedAvg":
      if args.dataset == "mnist":
         dataset_name = 'mnist'
         trainset = torchvision.datasets.MNIST(root='../data/'+dataset_name, train=True, download=True, transform=torchvision.transforms.Compose([
                           torchvision.transforms.ToTensor(),
                           torchvision.transforms.Normalize(
                              (0.1307,), (0.3081,))
                        ]))
         testset = torchvision.datasets.MNIST(root='../data'+dataset_name, train=False, download=True, transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                          (0.1307,), (0.3081,))
                                    ]))
         if args.iid == "true":
            #construct an iid mnist dataset.
            #distribute data among clients
            client_data_dict = {}
            all_indices = np.arange(0,len(trainset))
            available_indices = np.arange(len(trainset))
            for client_idx in range(args.K):
               selected_indices = np.random.choice(available_indices, 600, replace=False)
               client_data_dict[client_idx] = selected_indices
               available_indices = np.setdiff1d(available_indices, selected_indices)
            # fed_avg(args, client_data_dict, trainset, testset)
            if args.gpu == "gpu":
               device = torch.device('cuda:0')
            else:
               device = torch.device('cpu')
            
            if args.model == 'nn':
               model_global = models.MP(28*28,200,10)
               model_global.to(device)
               model_local = models.MP(28*28,200,10)
               criterion = torch.nn.CrossEntropyLoss()
            
            if args.model == 'cnn':
               model_global = models.CNN_MNIST()
               model_global.to(device)
               model_local = models.CNN_MNIST()
               criterion = torch.nn.CrossEntropyLoss()
               
            w_global = model_global.state_dict()
            num_samples_dict = {} #number of samples per user.

            for i in range (args.K): #loop through clients
               num_samples_dict[i] = len(client_data_dict[i]) 
            best_test_acc = -1
            test_acc = []
            for rounds in range(args.T): #total number of rounds
               if args.C == 0:
                  m = 1 
               else:
                  m = int(max(args.C*args.K, 1)) 
               
               client_indices = np.random.choice(args.K, m, replace=False)
               #client_indices.astype(int)
               num_samples_list = [num_samples_dict[idx] for idx in client_indices] #list containing number of samples for each user id.
               total_num_samples = reduce(lambda x,y: x+y, num_samples_list, 0)
               store = {}

               ##
               # Start all threads. 
               threads = []
               for i in range(len(client_indices)):
                  model_local = models.MP(28*28,200,10)
                  t = Thread(target=client_update, args=(trainset, i, client_idx, w_global.copy(), args, client_data_dict, criterion, model_local,))
                  t.start()
                  threads.append(t)

               # Wait all threads to finish.
               for t in threads:
                  t.join()
                              
               # for index,client_idx in enumerate(client_indices): #loop through selected clients
               #    w_local = client_update(trainset, client_idx, w_global.copy(), args, client_data_dict, criterion, model_local) #client index, global weight, args, dictionary of clients' data, criterion, optimizer.
               #    store[index] = w_local

               w_global = {}
               for layer in store[0]:
                  sum = 0
                  for user_key in store:
                     sum += store[user_key][layer]*num_samples_list[user_key]/total_num_samples
                  w_global[layer] = sum

               #Performing evaluation on test data.
               model_global.load_state_dict(w_global)
               test_loader = torch.utils.data.DataLoader(testset, batch_size=64,
                                          shuffle=False)
               with torch.no_grad():
                  model_global.eval()
                  running_loss = 0.0
                  running_acc = 0.0
                  for index,data in enumerate(test_loader):  
                     inputs, labels = data
                     inputs = inputs.to(device)
                     if args.model == 'nn':
                        inputs = inputs.flatten(1)
                     labels = labels.to(device)
                     output = model_global(inputs)
                     loss = criterion(output, labels)
                     pred = torch.argmax(output, dim=1)
                     acc = utils.accuracy(pred, labels)
                     running_acc += acc
                     running_loss += loss
               avg_acc = running_acc / (index+1)
               if best_test_acc<avg_acc:
                  best_test_acc = avg_acc
                  best_dict = {rounds: best_test_acc}
                  #torch.save(best_dict,'nn-trial.pt')
               print('Round '+ str(rounds))
               print(f'server stats: [loss: {running_loss / (index+1):.3f}')
               print(f'server stats: [accuracy: {running_acc / (index+1):.3f}')
               test_acc.append(running_acc / (index+1))
            # # torch.save(test_acc,'../stats/'+args.name)

         else:
            #construct a non-iid mnist dataset.
            #distribute data among clients
            client_data_dict = {}
            labels = trainset.targets.numpy()
            sorted_indices = np.argsort(labels)

            all_indices = np.arange(0,200)
            available_indices = np.arange(0,200)
            for client_idx in range(args.K):
               selected_indices = np.random.choice(available_indices, 2, replace=False)               
               A = sorted_indices[selected_indices[0]*300:selected_indices[0]*300+300]
               B = sorted_indices[selected_indices[1]*300:selected_indices[1]*300+300]
               merged_shards = np.concatenate((np.expand_dims(A, 0), np.expand_dims(B,0)), axis=1)
               client_data_dict[client_idx] = merged_shards[0]
               available_indices = np.setdiff1d(available_indices, selected_indices)

            # Test if non-iid data is ok.
            # for index in range(10):
            #    print(labels[client_data_dict[index]])
          
            #fed_avg(args, client_data_dict, trainset, testset)
         #fed_avg(args, client_data_dict, trainset, testset)




  
   
