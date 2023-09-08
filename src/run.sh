#!/bin/sh

#iid and mnist

#nn
python main.py --algo="FedAvg" --K=100 --C=0.1 --E=5 --B=10 --T=3 --lr=0.01 --gpu="gpu" --model="nn" > ../logs/exp11

#cnn
python main.py --algo="FedAvg" --K=100 --C=0.1 --E=5 --B=10 --T=3 --lr=0.01 --gpu="gpu" --model="cnn" > ../logs/exp12