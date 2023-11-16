from collections import namedtuple, OrderedDict
from typing import List, Dict, Tuple
import os
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn

import copy

import sys
import imagen as ig
import numpy as np
import numbergen as ng
import matplotlib.pyplot as plt
import matplotlib
import random
from PIL import Image
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torchvision as tv

import time
from typing import List, Tuple, Any, Dict
import torch.utils.data as data
import math


from sys import getsizeof, stderr
from itertools import chain

from collections import deque
from decimal import Decimal


import torchvision
import torchvision.transforms as transforms
import numpy as np




def generate_mpattern(x_input, y_input, num_class, num_picures):
    x_pattern = int(x_input * 2 / 3. - 1)
    y_pattern = int(y_input * 2 / 3. - 1)

    for cls in range(num_class):
        # define patterns
        patterns = []
        patterns.append(
            ig.Line(xdensity=x_pattern, ydensity=y_pattern, thickness=0.001, orientation=np.pi * ng.UniformRandom(),
                    x=ng.UniformRandom() - 0.5, y=ng.UniformRandom() - 0.5, scale=0.8))
        patterns.append(
            ig.Arc(xdensity=x_pattern, ydensity=y_pattern, thickness=0.001, orientation=np.pi * ng.UniformRandom(),
                    x=ng.UniformRandom() - 0.5, y=ng.UniformRandom() - 0.5, size=0.33))

        pat = np.zeros((x_pattern, y_pattern))
        for i in range(6):
            j = np.random.randint(len(patterns))
            pat += patterns[j]()
        res = pat > 0.5
        pat = res.astype(int)
        print(pat)

        x_offset = np.random.randint(x_input - x_pattern + 1)
        y_offset = np.random.randint(y_input - y_pattern + 1)
        print(x_offset, y_offset)

        for i in range(num_picures):
            base = np.random.rand(x_input, y_input)
            base[x_offset:x_offset + pat.shape[0], y_offset:y_offset + pat.shape[1]] += pat
            d = np.ones((x_input, x_input))
            img = np.minimum(base, d)
            # if not os.path.exists("./data/datasets/MPATTERN/" + str(cls) + "/"):
            #     os.makedirs("./data/datasets/MPATTERN/" + str(cls) + "/")
            plt.imsave("../data/datasets/MPATTERN/" + str(cls) + "/wm_" + str(i + 1) + ".png", img, cmap=matplotlib.cm.gray)


dataset_name = 'mnist'
trainset = torchvision.datasets.MNIST(root='../data/'+dataset_name, train=True, download=True, transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.1307,), (0.3081,))]))
print(trainset[0][0][0].shape)
print(trainset[0][1])
generate_mpattern(28,28,10,100)