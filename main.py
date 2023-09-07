# import sys
# from models import MP,CNN_CIFAR
# import torch
# import torchvision
# import torchvision.transforms as transforms
# import torch.nn as nn
# import tqdm
# import torch.optim as optim
# from options import arg_parser
# from utils import accuracy


# if __name__=='__main__':

#    args = arg_parser()
#    if args.dataset == "mnist":
#       dataset_name = 'mnist'
#       train_loader = torch.utils.data.DataLoader(
#       torchvision.datasets.MNIST('../data/'+dataset_name, train=True, download=True,
#                                  transform=torchvision.transforms.Compose([
#                                     torchvision.transforms.ToTensor(),
#                                     torchvision.transforms.Normalize(
#                                        (0.1307,), (0.3081,))
#                                  ])),
#       batch_size=64, shuffle=True)

#       test_loader = torch.utils.data.DataLoader(
#       torchvision.datasets.MNIST('../data/'+dataset_name, train=False, download=True,
#                                  transform=torchvision.transforms.Compose([
#                                     torchvision.transforms.ToTensor(),
#                                     torchvision.transforms.Normalize(
#                                        (0.1307,), (0.3081,))
#                                  ])),
#       batch_size=64, shuffle=True)
#       model = MP(784,200,10)
   
#    elif args.dataset == "CIFAR":
#       dataset_name = 'CIFAR'
#       transform = transforms.Compose(
#          [transforms.ToTensor(),
#          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#       batch_size = 64
#       trainset = torchvision.datasets.CIFAR10(root='../data/'+dataset_name, train=True,
#                                              download=True, transform=transform)
#       train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
#                                                 shuffle=True, num_workers=2)
#       testset = torchvision.datasets.CIFAR10(root='../data/'+dataset_name, train=False,
#                                              download=True, transform=transform)
#       test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
#                                              shuffle=False, num_workers=2)
#       model = CNN_CIFAR()
   
#    if args.gpu == "gpu":
#       device = torch.device('cuda:0')
#    else:
#       device = torch.device('cpu')
   
#    model.to(device)
#    criterion = nn.CrossEntropyLoss()
#    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
#    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#    for epoch in range(30):  # loop over the dataset multiple times

#       running_loss = 0.0
#       running_acc = 0.0
#       for i, data in enumerate(train_loader, 0):
#          # get the inputs; data is a list of [inputs, labels]
#          inputs, labels = data
#          inputs = inputs.to(device)
#          labels = labels.to(device)
#          # zero the parameter gradients
#          optimizer.zero_grad()

#          # forward + backward + optimize

#          # inputs = inputs.flatten(1)
#          outputs = model(inputs)
#          pred = torch.argmax(outputs, dim=1)
#          loss = criterion(outputs, labels)
#          loss.backward()
#          optimizer.step()

#          # print statistics
#          acc = accuracy(pred, labels)
#          running_acc += acc
#          running_loss += loss.item()
#          if i % 100 == 99:    # print every 2000 mini-batches
#                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
#                print(f'[{epoch + 1}, {i + 1:5d}] accuracy: {running_acc / 100:.3f}')
#                running_loss = 0.0
#                running_acc = 0.0

#    print('Finished Training')
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
         if args.iid == "true":
            #construct an iid mnist dataset.
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
            #distribute data among clients
            client_data_dict = {}
            all_indices = np.arange(0,len(trainset))
            available_indices = np.arange(len(trainset))
            for client_idx in range(args.K):
               selected_indices = np.random.choice(available_indices, 600, replace=False)
               client_data_dict[client_idx] = selected_indices
               available_indices = np.setdiff1d(available_indices, selected_indices)
          
            fed_avg(args, client_data_dict, trainset, testset)




  
   
