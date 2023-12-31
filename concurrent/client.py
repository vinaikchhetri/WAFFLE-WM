from torch.utils.data import Dataset
import torch
import models
import torch.optim as optim
from torchvision.models import resnet18
from torchvision.models import vgg16
import torch.nn.utils.prune as prune

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

        return img, label

class Client():
    def __init__(self, data_client, trainset, args, device, is_adversary):
        self.is_adversary = is_adversary
        self.data_client = data_client
        self.args = args
        self.device = device
        self.trainset = trainset
        self.loss = None
        self.cd = CustomDataset(self.trainset, self.data_client)
        if args.B == 8:
            self.bs = len(trainset)
        else:
            self.bs = args.B
        self.data_loader = torch.utils.data.DataLoader(self.cd, batch_size=self.bs,
                                                shuffle=True)
        if args.gpu == "gpu":
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        
        if args.model == 'nn':
            self.model_local = models.MP(28*28,200,10)
            self.criterion = torch.nn.CrossEntropyLoss()
            self.model_local.to(device)
         
        if args.model == 'cnn':
            self.model_local = models.CNN_MNIST()
            self.criterion = torch.nn.CrossEntropyLoss()
            self.model_local.to(device)
            self.optimizer = optim.SGD(self.model_local.parameters(), lr=self.args.lr, momentum=0.5)
        
        if args.model == 'resnet':
            # self.model_local = resnet18(num_classes=10)
            if self.args.dataset == "cifar-10":
                self.model_local = models.ResNet(18, num_classes = 10)
            else:
                self.model_local = models.ResNet(18, num_classes = 100)
            self.criterion = torch.nn.CrossEntropyLoss()
            self.model_local.to(device)
            self.optimizer = optim.SGD(self.model_local.parameters(), lr=self.args.lr, momentum=0.5)

        if args.model == 'MNIST_L5':
            self.model_local = models.MNIST_L5()
            self.criterion = torch.nn.CrossEntropyLoss()
            self.model_local.to(device)
            self.optimizer = optim.SGD(self.model_local.parameters(), lr=self.args.lr, momentum=0.5)

        if args.model == 'vgg':
            self.model_local = vgg16(pretrained=True, progress=True)
            self.criterion = torch.nn.CrossEntropyLoss()
            self.model_local.to(device)
            self.optimizer = optim.SGD(self.model_local.parameters(), lr=self.args.lr, momentum=0.5)
            
    def load_model(self, model_global):
        self.model_local.load_state_dict(model_global.state_dict())

    def client_update(self):
        self.model_local.to(self.device)
        self.model_local.train()
        self.optimizer = optim.SGD(self.model_local.parameters(), lr=self.args.lr, momentum=0.5)
        if self.is_adversary == 0: # If not adversary do not finetune.
            epochs = self.args.E
        else:
            if self.args.finetune>0: # If adversary and we want to finetune then finetune.
                epochs = self.args.E + 50
            else: # If adversary and we only want to prune then no finetuning required.
                epochs = self.args.E

            if self.args.prune>0: # If adversary wants to prune. 
                for _, module in self.model_local.named_modules():
                    if isinstance(module, torch.nn.Conv2d):
                        prune.l1_unstructured(module, name='weight', amount=self.args.prune)
                        prune.remove(module, "weight")
                        
                    elif isinstance(module, torch.nn.Linear):
                        prune.l1_unstructured(module, name="weight", amount=self.args.prune)
                        prune.remove(module, "weight")
        
        for epoch in range(epochs):
            running_loss = 0.0
            running_acc = 0.0
            for index,data in enumerate(self.data_loader):
                
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                if self.args.model == 'nn':
                    inputs = inputs.flatten(1)
                outputs = self.model_local(inputs)
                pred = torch.argmax(outputs, dim=1)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
        return self.model_local