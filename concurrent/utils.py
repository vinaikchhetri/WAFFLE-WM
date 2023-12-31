import torch

def accuracy(predictions, labels):
    accuracy = 0
    for i,j in enumerate (predictions):
        if j == labels[i]:
            accuracy+=1
    return 100*accuracy/len(labels)


