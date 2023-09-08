#!/bin/sh

#iid and mnist

#nn

#B=inf
python main.py --algo="FedAvg" --K=100 --C=0 --E=1 --B=8 --T=1500 --lr=0.01 --gpu="gpu" --model="nn" > ../logs/exp_nn_inf_0.txt
python main.py --algo="FedAvg" --K=100 --C=0.1 --E=1 --B=8 --T=1500 --lr=0.01 --gpu="gpu" --model="nn" > ../logs/exp_nn_inf_0.1.txt
python main.py --algo="FedAvg" --K=100 --C=0.2 --E=1 --B=8 --T=1500 --lr=0.01 --gpu="gpu" --model="nn" > ../logs/exp_nn_inf_0.2.txt

#B=10
python main.py --algo="FedAvg" --K=100 --C=0 --E=1 --B=10 --T=400 --lr=0.01 --gpu="gpu" --model="nn" > ../logs/exp_nn_10_0.txt
python main.py --algo="FedAvg" --K=100 --C=0.1 --E=1 --B=10 --T=400 --lr=0.01 --gpu="gpu" --model="nn" > ../logs/exp_nn_10_0.1.txt
python main.py --algo="FedAvg" --K=100 --C=0.2 --E=1 --B=10 --T=400 --lr=0.01 --gpu="gpu" --model="nn" > ../logs/exp_nn_10_0.2.txt
python main.py --algo="FedAvg" --K=100 --C=0.5 --E=1 --B=10 --T=400 --lr=0.01 --gpu="gpu" --model="nn" > ../logs/exp_nn_10_0.5.txt
python main.py --algo="FedAvg" --K=100 --C=1 --E=1 --B=10 --T=400 --lr=0.01 --gpu="gpu" --model="nn" > ../logs/exp_nn_10_1.txt

#cnn

#B=inf
python main.py --algo="FedAvg" --K=100 --C=0 --E=5--B=8 --T=400 --lr=0.01 --gpu="gpu" --model="cnn" > ../logs/exp_cnn_inf_0.txt
python main.py --algo="FedAvg" --K=100 --C=0.1 --E=5 --B=8 --T=400 --lr=0.01 --gpu="gpu" --model="cnn" > ../logs/exp_cnn_inf_0.1.txt
python main.py --algo="FedAvg" --K=100 --C=0.2 --E=5 --B=8 --T=400 --lr=0.01 --gpu="gpu" --model="cnn" > ../logs/exp_cnn_inf_0.2.txt
python main.py --algo="FedAvg" --K=100 --C=0.5 --E=5 --B=8 --T=400 --lr=0.01 --gpu="gpu" --model="cnn" > ../logs/exp_cnn_inf_0.5.txt
python main.py --algo="FedAvg" --K=100 --C=1 --E=5 --B=8 --T=400 --lr=0.01 --gpu="gpu" --model="cnn" > ../logs/exp_cnn_inf_1.txt

#B=10
python main.py --algo="FedAvg" --K=100 --C=0 --E=5--B=10 --T=100 --lr=0.01 --gpu="gpu" --model="cnn" > ../logs/exp_cnn_10_0.txt
python main.py --algo="FedAvg" --K=100 --C=0.1 --E=5 --B=10 --T=100 --lr=0.01 --gpu="gpu" --model="cnn" > ../logs/exp_cnn_10_0.1.txt
python main.py --algo="FedAvg" --K=100 --C=0.2 --E=5 --B=10 --T=100 --lr=0.01 --gpu="gpu" --model="cnn" > ../logs/exp_cnn_10_0.2.txt
python main.py --algo="FedAvg" --K=100 --C=0.5 --E=5 --B=10 --T=100 --lr=0.01 --gpu="gpu" --model="cnn" > ../logs/exp_cnn_10_0.5.txt
python main.py --algo="FedAvg" --K=100 --C=1 --E=5 --B=10 --T=100 --lr=0.01 --gpu="gpu" --model="cnn" > ../logs/exp_cnn_10_1.txt
