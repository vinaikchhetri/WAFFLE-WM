#!/bin/sh

# MNIST
# non-compromised
python3 main.py --algo="FedAvg" --dataset 'mnist' --K=100 --C=0.1 --E=5 --B=100 --T=10 --lr=0.1 --gpu="gpu" --model="cnn" --retrainingR=100 --dataset="mnist" --iid="true" --pretrainingR=25 --num_attackers=0 --finetune=0 --prune=0 --name="exp_mnist_a0_f0_p0"
# 80 attackers and 70% prune rate
python3 main.py --algo="FedAvg" --dataset 'mnist' --K=100 --C=0.1 --E=5 --B=100 --T=10 --lr=0.1 --gpu="gpu" --model="cnn" --retrainingR=100 --dataset="mnist" --iid="true" --pretrainingR=25 --num_attackers=80 --finetune=0 --prune=0.7 --name="exp_mnist_a80_f0_p70"
# 80 attackers with 70% prune rate and 50 extra expochs of finetuning
python3 main.py --algo="FedAvg" --dataset 'mnist' --K=100 --C=0.1 --E=5 --B=100 --T=10 --lr=0.1 --gpu="gpu" --model="cnn" --retrainingR=100 --dataset="mnist" --iid="true" --pretrainingR=25 --num_attackers=80 --finetune=50 --prune=0.7 --name="exp_mnist_a80_f50_p70"

# CIFAR-10
# non-compromised
python3 main.py --algo="FedAvg" --dataset 'cifar-10' --K=100 --C=0.1 --E=10 --B=100 --T=30 --lr=0.01 --gpu="gpu" --model="resnet" --retrainingR=100 --dataset="cifar-10" --iid="true" --pretrainingR=30 --num_attackers=0 --finetune=0 --prune=0 --name="exp_cifart_a0_f0_p0"
# 80 attackers and 70% prune rate
python3 main.py --algo="FedAvg" --dataset 'cifar-10' --K=100 --C=0.1 --E=10 --B=100 --T=30 --lr=0.01 --gpu="gpu" --model="resnet" --retrainingR=100 --dataset="cifar-10" --iid="true" --pretrainingR=30 --num_attackers=80 --finetune=0 --prune=0.7 --name="exp_cifart_a80_f0_p70"
# 80 attackers with 70% prune rate and 50 extra expochs of finetuning
python3 main.py --algo="FedAvg" --dataset 'cifar-10' --K=100 --C=0.1 --E=10 --B=100 --T=30 --lr=0.01 --gpu="gpu" --model="resnet" --retrainingR=100 --dataset="cifar-10" --iid="true" --pretrainingR=30 --num_attackers=80 --finetune=50 --prune=0.7 --name="exp_cifart_a80_f50_p70"

# CIFAR-100
# non-compromised
python3 main.py --algo="FedAvg" --dataset 'cifar-100' --K=100 --C=0.1 --E=10 --B=100 --T=50 --lr=0.01 --gpu="gpu" --model="resnet" --retrainingR=100 --dataset="cifar-100" --iid="true" --pretrainingR=25 --num_attackers=0 --finetune=0 --prune=0 --name="exp_cifar100_a0_f0_p0"
# 80 attackers and 70% prune rate
python3 main.py --algo="FedAvg" --dataset 'cifar-100' --K=100 --C=0.1 --E=10 --B=100 --T=50 --lr=0.01 --gpu="gpu" --model="resnet" --retrainingR=100 --dataset="cifar-100" --iid="true" --pretrainingR=25 --num_attackers=80 --finetune=0 --prune=0.7 --name="exp_cifar100_a80_f0_p70"
# 80 attackers with 70% prune rate and 50 extra expochs of finetuning
python3 main.py --algo="FedAvg" --dataset 'cifar-100' --K=100 --C=0.1 --E=10 --B=100 --T=50 --lr=0.01 --gpu="gpu" --model="resnet" --retrainingR=100 --dataset="cifar-100" --iid="true" --pretrainingR=25 --num_attackers=80 --finetune=50 --prune=0.7 --name="exp_cifar100_a80_f50_p70"


