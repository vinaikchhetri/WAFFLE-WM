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
import server
from torchvision.models import resnet18
import data_splitter

if __name__=='__main__':
   args = arg_parser()
   serv = server.Server(args) # Initialise the server.
   serv.create_clients() # Create clients.
   serv.train() # Begin training.
   
