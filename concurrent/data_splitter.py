import torchvision
import torchvision.transforms as transforms
import numpy as np
import watermark
import data_handle
import argparse

def splitter(args):
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
            # generate train=test watermark dataset loader.
            mean = [0.5]#[0.1307]
            std = [0.5]#[0.3081]
            greyscale = [torchvision.transforms.Grayscale()] 
            watermark_transforms = torchvision.transforms.Compose(greyscale + [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean, std)
                ])
            watermarkset = data_handle.Pattern(root_dir='../data/datasets/MPATTERN/' , train= True, transform=watermark_transforms , download= True)
            #print("$$$$$$$$$",len(watermarkset))

            
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
            mean = [0.5074,0.4867,0.4411]
            std  = [0.2011,0.1987,0.2025]

            watermark_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(32),
            torchvision.transforms.CenterCrop(32),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std)
        ])

            watermarkset = data_handle.Pattern(root_dir='../data/datasets/HPATTERN/' , train= True, transform=watermark_transforms , download= True)
            if args.iid == "true":
                #construct an iid mnist dataset.
                #distribute data among clients
                client_data_dict = {}
                all_indices = np.arange(0,len(trainset))
                available_indices = np.arange(len(trainset))
                for client_idx in range(args.K):
                    selected_indices = np.random.choice(available_indices, 500, replace=False)
                    client_data_dict[client_idx] = selected_indices
                    available_indices = np.setdiff1d(available_indices, selected_indices)
            
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
                    client_data_dict[client_idx] = merged_shards[0].astype(int)
                    available_indices = np.setdiff1d(available_indices, selected_indices)

        if args.dataset == "cifar-10":
            dataset_name = 'cifar-10'
            train_data = torchvision.datasets.CIFAR10('./', train=True, download=True)
            transform = transforms.Compose(
                [transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            trainset = torchvision.datasets.CIFAR10(root='../data/'+dataset_name,
                                            train=True,
                                            download=True,
                                            transform=transform)

            testset = torchvision.datasets.CIFAR10(root='../data/'+dataset_name,
                                            train=False,
                                            download=True,
                                            transform=transform)

            # mean = [0.5, 0.5, 0.5]
            # std = [0.5, 0.5, 0.5]
            mean = [0.485, 0.456, 0.406]
            std  = [0.229, 0.224, 0.225]

            watermark_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(32),
            torchvision.transforms.CenterCrop(32),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std)
        ])
            watermarkset = data_handle.Pattern(root_dir='../data/datasets/CPATTERN/' , train= True, transform=watermark_transforms , download= True)
            if args.iid == "true":
                #construct an iid mnist dataset.
                #distribute data among clients
                client_data_dict = {}
                all_indices = np.arange(0,len(trainset))
                available_indices = np.arange(len(trainset))
                for client_idx in range(args.K):
                    selected_indices = np.random.choice(available_indices, 500, replace=False)
                    client_data_dict[client_idx] = selected_indices
                    available_indices = np.setdiff1d(available_indices, selected_indices)
        
    
        return trainset,testset,client_data_dict,watermarkset

            
# parser = argparse.ArgumentParser()
# parser.add_argument(algo="FedAvg", K=100, C=0, E=1, B=8, T=1500, lr=0.01, gpu="gpu", model="nn")
# args = parser.parse_args()

# parser = argparse.ArgumentParser(description='Your program description')

# # Add arguments
# parser.add_argument('-algo', type=str, default="FedAvg", help='Description of argument k')
# parser.add_argument('--K', type=int, default=100, help='Description of argument name')
# parser.add_argument('--C', type=int, default=0.2, help='Description of argument name')
# parser.add_argument('--e', type=int, default=1, help='Description of argument name')
# parser.add_argument('--B', type=int, default=8, help='Description of argument name')
# parser.add_argument('--T', type=int, default=1500, help='Description of argument name')
# parser.add_argument('--lr', type=float, default=0.01, help='Description of argument name')
# parser.add_argument('--gpu', type=str, default="gpu", help='Description of argument name')
# parser.add_argument('--model', type=str, default="nn", help='Description of argument name')
# parser.add_argument('--dataset', type=str, default="mnist", help='Description of argument name')
# parser.add_argument('--iid', type=str, default="true", help='Description of argument name')

# # Parse the command-line arguments
# args = parser.parse_args()
# splitter(args)
