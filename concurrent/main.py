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
from utils import accuracy
from fedavg import fed_avg


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
         fed_avg(args, client_data_dict, trainset, testset)




  
   
