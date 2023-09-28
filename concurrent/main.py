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
import time

import concurrent.futures
import os
import client
from torchvision.models import resnet18

print(os.cpu_count())

#store = {}

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

def client_update(client):
   
   client.model_local.to(client.device)
   client.model_local.train()
   client.optimizer = optim.SGD(client.model_local.parameters(), lr=0.01, momentum=0.5)
   for epoch in range(client.args.E):
      running_loss = 0.0
      running_acc = 0.0
      for index,data in enumerate(client.data_loader):
         
         inputs, labels = data
         #print("inputs ", inputs.shape)
         #print("labels ", labels)
         inputs = inputs.to(client.device)
         labels = labels.to(client.device)

         client.optimizer.zero_grad()
         # for param in client.model_local.parameters():
         #    param.grad = None
         if args.model == 'nn':
            inputs = inputs.flatten(1)
         outputs = client.model_local(inputs)
         pred = torch.argmax(outputs, dim=1)
         loss = client.criterion(outputs, labels)
         #initial = time.time()
         loss.backward()
         #print(client_idx, time.time()-initial)
         client.optimizer.step()

   #return client.model_local.state_dict()
   return client.model_local


if __name__=='__main__':
   args = arg_parser()
   clients = []
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

         if args.gpu == "gpu":
            device = torch.device('cuda')
         else:
            device = torch.device('cpu')
         
         if args.model == 'nn':
            model_global = models.MP(28*28,200,10)
            model_global.to(device)
            #model_local = models.MP(28*28,200,10)
            criterion = torch.nn.CrossEntropyLoss()
         
         if args.model == 'cnn':
            # model_global = models.CNN_MNIST()
            model_global = models.CNN_MNIST()
            model_global.to(device)
            #model_local = models.CNN_MNIST()
            criterion = torch.nn.CrossEntropyLoss()
            
         w_global = model_global.state_dict()
         num_samples_dict = {} #number of samples per user.

         for i in range (args.K): #loop through clients
            num_samples_dict[i] = len(client_data_dict[i]) 
            clients.append(client.Client(client_data_dict[i], trainset, args, device))

         best_test_acc = -1
         test_acc = []
         #ini = time.time()


         initial = time.time()
         for rounds in range(args.T): #total number of rounds
            
            
            if args.C == 0:
               m = 1 
            else:
               m = int(max(args.C*args.K, 1)) 
            
            client_indices = np.random.choice(args.K, m, replace=False)
            client_indices.astype(int)
            num_samples_list = [num_samples_dict[idx] for idx in client_indices] #list containing number of samples for each user id.
            total_num_samples = reduce(lambda x,y: x+y, num_samples_list, 0)
            store = {}

            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(client_indices), os.cpu_count() - 1)) as executor:
               results = []
               # Submit tasks to the executor
               for index,client_idx in enumerate(client_indices): #loop through selected clients
                 # pars = (trainset, index, client_idx, w_global.copy(), args, client_data_dict, criterion, model_local)
                  # initial = time.time()
                  #clients[client_idx].load_model(w_global)
                  clients[client_idx].load_model(model_global)
                  results.append(executor.submit(client_update, clients[client_idx]).result())
                  #results.append(executor.submit(client_update, clients[client_idx]))
                  #print(index," - ",time.time()-initial)
                  
               # Retrieve results as they become available
               for ind,future in enumerate(results):
                  store[ind] = future.state_dict()


  

            # # #print("finished ", time.time() - initial)
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
            test_acc = []
            test_loss = []
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
            # avg_acc = running_acc / (index+1)
            # if best_test_acc<avg_acc:
            #    best_test_acc = avg_acc
            #    best_dict = {rounds: best_test_acc}
               #torch.save(best_dict,'nn-trial.pt')
            print('Round '+ str(rounds))
            print(f'server stats: [loss: {running_loss / (index+1):.3f}')
            print(f'server stats: [accuracy: {running_acc / (index+1):.3f}')
            test_acc.append(running_acc / (index+1))
            test_loss.append(running_loss / (index+1))
         torch.save(test_acc,'../new_stats/mnist/'+'acc-'+args.name)
         torch.save(test_loss,'../new_stats/mnist/'+'loss-'+args.name)
         # torch.save(test_acc,'../new_stats/mnist/'+'acc-'+args.name)
        
         print("finished ", time.time() - initial)


      if args.dataset == "cifar-100":
         dataset_name = 'cifar-100'
         train_data = torchvision.datasets.CIFAR100('./', train=True, download=True)

         transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
         trainset = torchvision.datasets.CIFAR100(root='../data/'+dataset_name,
                                         train=True,
                                         download=True,
                                         transform=transform)

         testset = torchvision.datasets.CIFAR100(root='../data/'+dataset_name,
                                        train=False,
                                        download=True,
                                        transform=transform)


         if args.iid == "true":
            #construct an iid mnist dataset.
            #distribute data among clients
            client_data_dict = {}
            all_indices = np.arange(0,len(trainset))
            available_indices = np.arange(len(trainset))
            for client_idx in range(args.K):
               selected_indices = np.random.choice(available_indices, 500, replace=False)
               client_data_dict[client_idx] = selected_indices
               #print(client_data_dict[client_idx])
               #print(len(client_data_dict[client_idx]))
               #break
               available_indices = np.setdiff1d(available_indices, selected_indices)
            # fed_avg(args, client_data_dict, trainset, testset)
         else:
            #construct a non-iid mnist dataset.
            #distribute data among clients
            client_data_dict = {}
            
            labels = np.asarray(trainset.targets)
            sorted_indices = np.argsort(labels)

            all_indices = np.arange(0,1000)
            available_indices = np.arange(0,1000)
            
            for client_idx in range(args.K):
               merged_shards = np.array([[]])
               selected_indices = np.random.choice(available_indices, 10, replace=False)
               for index in range(10):
                  temp = sorted_indices[selected_indices[index]*50:selected_indices[index]*50+50]               
                  merged_shards = np.concatenate((merged_shards, np.expand_dims(temp,0)), axis=1)
               client_data_dict[client_idx] = merged_shards[0]
               #print(client_data_dict[client_idx])
               #break
               available_indices = np.setdiff1d(available_indices, selected_indices)
            
         if args.gpu == "gpu":
            device = torch.device('cuda')
         else:
            device = torch.device('cpu')

         model_global = resnet18(num_classes=100)
         model_global.to(device)
         criterion = torch.nn.CrossEntropyLoss()
         
            
         w_global = model_global.state_dict()
         num_samples_dict = {} #number of samples per user.

         for i in range (args.K): #loop through clients
            num_samples_dict[i] = len(client_data_dict[i]) 
       
            clients.append(client.Client(client_data_dict[i].astype(int), trainset, args, device))

         best_test_acc = -1
         test_acc = []
         #ini = time.time()


         initial = time.time()
         for rounds in range(args.T): #total number of rounds
            
            
            if args.C == 0:
               m = 1 
            else:
               m = int(max(args.C*args.K, 1)) 
            
            client_indices = np.random.choice(args.K, m, replace=False)
            client_indices.astype(int)
            num_samples_list = [num_samples_dict[idx] for idx in client_indices] #list containing number of samples for each user id.
            total_num_samples = reduce(lambda x,y: x+y, num_samples_list, 0)
            store = {}

            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(client_indices), os.cpu_count() - 1)) as executor:
               results = []
               # Submit tasks to the executor
               for index,client_idx in enumerate(client_indices): #loop through selected clients
                 # pars = (trainset, index, client_idx, w_global.copy(), args, client_data_dict, criterion, model_local)
                  # initial = time.time()
                  #clients[client_idx].load_model(w_global)
                  clients[client_idx].load_model(model_global)
                  results.append(executor.submit(client_update, clients[client_idx]).result())
                  #results.append(executor.submit(client_update, clients[client_idx]))
                  #print(index," - ",time.time()-initial)
                  
               # Retrieve results as they become available
               for ind,future in enumerate(results):
                  store[ind] = future.state_dict()


  

            # # #print("finished ", time.time() - initial)
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
            test_acc = []
            test_loss = []
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
            # avg_acc = running_acc / (index+1)
            # if best_test_acc<avg_acc:
            #    best_test_acc = avg_acc
            #    best_dict = {rounds: best_test_acc}
               #torch.save(best_dict,'nn-trial.pt')
            print('Round '+ str(rounds))
            print(f'server stats: [loss: {running_loss / (index+1):.3f}')
            print(f'server stats: [accuracy: {running_acc / (index+1):.3f}')
            test_acc.append(running_acc / (index+1))
            test_loss.append(running_loss / (index+1))
         #torch.save(test_acc,'../new_stats/mnist/'+'acc-'+args.name)
         #torch.save(test_loss,'../new_stats/mnist/'+'loss-'+args.name)
         # torch.save(test_acc,'../new_stats/mnist/'+'acc-'+args.name)
        
         print("finished ", time.time() - initial)




         # batch_size = 128
         # trainloader = torch.utils.data.DataLoader(
         #    trainset, batch_size, shuffle=True, num_workers=2,pin_memory=True)

         
         # testloader = torch.utils.data.DataLoader(
         #    testset, batch_size*2,pin_memory=True, num_workers=2)
         
         # net = resnet18(num_classes=100)
         # criterion = nn.CrossEntropyLoss()
         # optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

         # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
         # net.to(device)

         # num_epochs = 100

         # for epoch in range(num_epochs):
         #    net.train()
         #    running_loss = 0.0
         #    correct = 0
         #    for i, data in enumerate(trainloader, 0):
         #       inputs, labels = data
         #       inputs, labels = inputs.to(device), labels.to(device)

         #       optimizer.zero_grad()

         #       outputs = net(inputs)
         #       loss = criterion(outputs, labels)
         #       loss.backward()
         #       optimizer.step()

         #       running_loss += loss.item()
         #       _, predicted = torch.max(outputs.data, 1)
         #       correct += (predicted == labels).sum().item()

         #    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}")
         #    print(f"Epoch {epoch + 1}, Acc: {correct / len(trainloader)}")

         # print("Finished Training")

         # net.eval()
         # correct = 0
         # total = 0

         # with torch.no_grad():
         #    for data in testloader:
         #       inputs, labels = data
         #       inputs, labels = inputs.to(device), labels.to(device)
         #       outputs = net(inputs)
         #       _, predicted = torch.max(outputs.data, 1)
         #       total += labels.size(0)
         #       correct += (predicted == labels).sum().item()

         # accuracy = 100 * correct / total
         # print(f"Accuracy on test set: {accuracy}%")


  
   
