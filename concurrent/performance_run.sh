#!/bin/sh

# MNIST

# normal
python3 main.py --algo="FedAvg" --dataset 'mnist' --K=100 --C=0.1 --E=5 --B=100 --T=10 --lr=0.1 --gpu="gpu" --model="cnn" --retrainingR=0 --dataset="mnist" --iid="true" --pretrainingR=0 --num_attackers=0 --finetune=0 --prune=0 --benchmark=1 --name="normal_exp_mnist_a0_f0_p0"


# waffle
python3 main.py --algo="FedAvg" --dataset 'mnist' --K=100 --C=0.1 --E=5 --B=100 --T=10 --lr=0.1 --gpu="gpu" --model="cnn" --retrainingR=100 --dataset="mnist" --iid="true" --pretrainingR=25 --num_attackers=0 --finetune=0 --prune=0 --benchmark=0 --name="waffle_exp_mnist_a0_f0_p0"


# CIFAR-10

# normal
python3 main.py --algo="FedAvg" --dataset 'cifar-10' --K=100 --C=0.1 --E=10 --B=100 --T=30 --lr=0.01 --gpu="gpu" --model="resnet" --retrainingR=0 --dataset="cifar-10" --iid="true" --pretrainingR=0 --num_attackers=0 --finetune=0 --prune=0 --benchmark=1 --name="normal_exp_cifart_a0_f0_p0"


# waffle
python3 main.py --algo="FedAvg" --dataset 'cifar-10' --K=100 --C=0.1 --E=10 --B=100 --T=30 --lr=0.01 --gpu="gpu" --model="resnet" --retrainingR=100 --dataset="cifar-10" --iid="true" --pretrainingR=30 --num_attackers=0 --finetune=0 --prune=0 --benchmark=0 --name="waffle_exp_cifart_a0_f0_p0"


# CIFAR-100

# normal
python3 main.py --algo="FedAvg" --dataset 'cifar-100' --K=100 --C=0.1 --E=10 --B=100 --T=50 --lr=0.01 --gpu="gpu" --model="resnet" --retrainingR=0 --dataset="cifar-100" --iid="true" --pretrainingR=0 --num_attackers=0 --finetune=0 --prune=0 --benchmark=1 --name="normal_exp_cifar100_a0_f0_p0"

# waffle
python3 main.py --algo="FedAvg" --dataset 'cifar-100' --K=100 --C=0.1 --E=10 --B=100 --T=50 --lr=0.01 --gpu="gpu" --model="resnet" --retrainingR=100 --dataset="cifar-100" --iid="true" --pretrainingR=25 --num_attackers=0 --finetune=0 --prune=0 --benchmark=0 --name="waffle_exp_cifar100_a0_f0_p0"




