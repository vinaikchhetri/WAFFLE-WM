#!/bin/sh

# #iid and mnist

# #nn

# python3 main.py --algo="FedAvg" --K=100 --C=0.1 --E=5 --B=10 --T=10 --lr=0.01 --gpu="gpu" --model="resnet" --retrainingR=5 --dataset="cifar-10" --iid="true" --pretrainingR=25 --name="asda_12"

# python3 main.py --algo="FedAvg" --K=100 --C=0.1 --E=5 --B=10 --T=10 --lr=0.01 --gpu="gpu" --model="vgg" --retrainingR=5 --dataset="mnist" --iid="true" --pretrainingR=25 --name="asda_12" > ../new_logs/mnist/"asdf"
# python3 main.py --algo="FedAvg" --K=100 --C=0.1 --E=5 --B=10 --T=10 --lr=0.01 --gpu="gpu" --model="nn" --retrainingR=5 --dataset="mnist" --iid="true" --pretrainingR=25 --name="asda_12"

# python main.py --algo="FedAvg" --dataset 'cifar-100' --K=100 --C=0.1 --E=$e --B=$b --T=1000 --lr=0.01 --gpu="gpu" --model="resnet" --iid="false"  --retrainingR=5
# python3 main.py --algo="FedAvg" --K=100 --C=0.1 --E=5 --B=10 --T=10 --lr=0.01 --gpu="gpu" --model="nn" --retrainingR=5

# #B=inf
# python main.py --algo="FedAvg" --K=100 --C=0 --E=1 --B=8 --T=1500 --lr=0.01 --gpu="gpu" --model="nn" --name="exp_nn_inf_0" > ../logs/exp_nn_inf_0.txt
# python main.py --algo="FedAvg" --K=100 --C=0.1 --E=1 --B=8 --T=1500 --lr=0.01 --gpu="gpu" --model="nn" --name="exp_nn_inf_0.1" > ../logs/exp_nn_inf_0.1.txt
# python main.py --algo="FedAvg" --K=100 --C=0.2 --E=1 --B=8 --T=1500 --lr=0.01 --gpu="gpu" --model="nn" --name="exp_nn_inf_0.2" > ../logs/exp_nn_inf_0.2.txt

# #B=10
# python main.py --algo="FedAvg" --K=100 --C=0 --E=1 --B=10 --T=400 --lr=0.01 --gpu="gpu" --model="nn" --name="exp_nn_10_0" > ../logs/exp_nn_10_0.txt
# python main.py --algo="FedAvg" --K=100 --C=0.1 --E=1 --B=10 --T=400 --lr=0.01 --gpu="gpu" --model="nn" --name="exp_nn_10_0.1" > ../logs/exp_nn_10_0.1.txt
# python main.py --algo="FedAvg" --K=100 --C=0.2 --E=1 --B=10 --T=400 --lr=0.01 --gpu="gpu" --model="nn" --name="exp_nn_10_0.2" > ../logs/exp_nn_10_0.2.txt
# python main.py --algo="FedAvg" --K=100 --C=0.5 --E=1 --B=10 --T=400 --lr=0.01 --gpu="gpu" --model="nn" --name="exp_nn_10_0.5" > ../logs/exp_nn_10_0.5.txt
# python main.py --algo="FedAvg" --K=100 --C=1 --E=1 --B=10 --T=400 --lr=0.01 --gpu="gpu" --model="nn" --name="exp_nn_10_1" > ../logs/exp_nn_10_1.txt

#cnn

#B=inf
#python main.py --algo="FedAvg" --K=100 --C=0 --E=5 --B=8 --T=400 --lr=0.01 --gpu="gpu" --model="cnn" --name="exp_cnn_inf_0" > ../logs/exp_cnn_inf_0.txt
#python main.py --algo="FedAvg" --K=100 --C=0.1 --E=5 --B=8 --T=400 --lr=0.01 --gpu="gpu" --model="cnn" --name="exp_cnn_inf_0.1" > ../logs/exp_cnn_inf_0.1.txt
#python main.py --algo="FedAvg" --K=100 --C=0.2 --E=5 --B=8 --T=400 --lr=0.01 --gpu="gpu" --model="cnn" --name="exp_cnn_inf_0.2" > ../logs/exp_cnn_inf_0.2.txt
#python main.py --algo="FedAvg" --K=100 --C=0.5 --E=5 --B=8 --T=400 --lr=0.01 --gpu="gpu" --model="cnn" --name="exp_cnn_inf_0.5" > ../logs/exp_cnn_inf_0.5.txt
#python main.py --algo="FedAvg" --K=100 --C=1 --E=5 --B=8 --T=400 --lr=0.01 --gpu="gpu" --model="cnn" --name="exp_cnn_inf_1" > ../logs/exp_cnn_inf_1.txt

#B=10
#python main.py --algo="FedAvg" --K=100 --C=0 --E=5 --B=10 --T=100 --lr=0.01 --gpu="gpu" --model="cnn" --name="exp_cnn_10_0" > ../logs/exp_cnn_10_0.txt
#python main.py --algo="FedAvg" --K=100 --C=0.1 --E=5 --B=10 --T=100 --lr=0.01 --gpu="gpu" --model="cnn" --name="exp_cnn_10_0.1" > ../logs/exp_cnn_10_0.1.txt
#python main.py --algo="FedAvg" --K=100 --C=0.2 --E=5 --B=10 --T=100 --lr=0.01 --gpu="gpu" --model="cnn" --name="exp_cnn_10_0.2" > ../logs/exp_cnn_10_0.2.txt
#python main.py --algo="FedAvg" --K=100 --C=0.5 --E=5 --B=10 --T=100 --lr=0.01 --gpu="gpu" --model="cnn" --name="exp_cnn_10_0.5" > ../logs/exp_cnn_10_0.5.txt
#python main.py --algo="FedAvg" --K=100 --C=1 --E=5 --B=10 --T=100 --lr=0.01 --gpu="gpu" --model="cnn" --name="exp_cnn_10_1" > ../logs/exp_cnn_10_1.txt

# non-iid and mnist

# nn

# B=inf
# python main.py --algo="FedAvg" --K=100 --C=0 --E=1 --B=8 --T=5000 --lr=0.01 --gpu="gpu" --model="nn" --iid="false" --name="exp_nn_inf_0_f" > ../logs/exp_nn_inf_0_f.txt
# python main.py --algo="FedAvg" --K=100 --C=0.1 --E=1 --B=8 --T=2000 --lr=0.01 --gpu="gpu" --model="nn" --iid="false" --name="exp_nn_inf_0.1_f" > ../logs/exp_nn_inf_0.1_f.txt
# python main.py --algo="FedAvg" --K=100 --C=0.2 --E=1 --B=8 --T=2000 --lr=0.01 --gpu="gpu" --model="nn" --iid="false" --name="exp_nn_inf_0.2_f" > ../logs/exp_nn_inf_0.2_f.txt

# #B=10
# python main.py --algo="FedAvg" --K=100 --C=0 --E=1 --B=10 --T=4000 --lr=0.01 --gpu="gpu" --model="nn" --iid="false" --name="exp_nn_10_0_f" > ../logs/exp_nn_10_0_f.txt
# python main.py --algo="FedAvg" --K=100 --C=0.1 --E=1 --B=10 --T=800 --lr=0.01 --gpu="gpu" --model="nn" --iid="false" --name="exp_nn_10_0.1_f" > ../logs/exp_nn_10_0.1_f.txt
# python main.py --algo="FedAvg" --K=100 --C=0.2 --E=1 --B=10 --T=800 --lr=0.01 --gpu="gpu" --model="nn" --iid="false" --name="exp_nn_10_0.2_f" > ../logs/exp_nn_10_0.2_f.txt
# python main.py --algo="FedAvg" --K=100 --C=0.5 --E=1 --B=10 --T=600 --lr=0.01 --gpu="gpu" --model="nn" --iid="false" --name="exp_nn_10_0.5_f" > ../logs/exp_nn_10_0.5_f.txt
# python main.py --algo="FedAvg" --K=100 --C=1 --E=1 --B=10 --T=600 --lr=0.01 --gpu="gpu" --model="nn" --iid="false" --name="exp_nn_10_1_f" > ../logs/exp_nn_10_1_f.txt

# #cnn

# #B=inf
# python main.py --algo="FedAvg" --K=100 --C=0 --E=5 --B=8 --T=1500 --lr=0.01 --gpu="gpu" --model="cnn" --iid="false" --name="exp_cnn_inf_0_f" > ../logs/exp_cnn_inf_0_f.txt
# python main.py --algo="FedAvg" --K=100 --C=0.1 --E=5 --B=8 --T=1500 --lr=0.01 --gpu="gpu" --model="cnn" --iid="false" --name="exp_cnn_inf_0.1_f" > ../logs/exp_cnn_inf_0.1_f.txt
# python main.py --algo="FedAvg" --K=100 --C=0.2 --E=5 --B=8 --T=1500 --lr=0.01 --gpu="gpu" --model="cnn" --iid="false" --name="exp_cnn_inf_0.2_f" > ../logs/exp_cnn_inf_0.2_f.txt
# python main.py --algo="FedAvg" --K=100 --C=0.5 --E=5 --B=8 --T=1500 --lr=0.01 --gpu="gpu" --model="cnn" --iid="false" --name="exp_cnn_inf_0.5_f" > ../logs/exp_cnn_inf_0.5_f.txt

# #B=10
# python main.py --algo="FedAvg" --K=100 --C=0 --E=5 --B=10 --T=1500 --lr=0.01 --gpu="gpu" --model="cnn" --iid="false" --name="exp_cnn_10_0_f" > ../logs/exp_cnn_10_0_f.txt
# python main.py --algo="FedAvg" --K=100 --C=0.1 --E=5 --B=10 --T=500 --lr=0.01 --gpu="gpu" --model="cnn" --iid="false" --name="exp_cnn_10_0.1_f" > ../logs/exp_cnn_10_0.1_f.txt
# python main.py --algo="FedAvg" --K=100 --C=0.2 --E=5 --B=10 --T=500 --lr=0.01 --gpu="gpu" --model="cnn" --iid="false" --name="exp_cnn_10_0.2_f" > ../logs/exp_cnn_10_0.2_f.txt
# python main.py --algo="FedAvg" --K=100 --C=0.5 --E=5 --B=10 --T=500 --lr=0.01 --gpu="gpu" --model="cnn" --iid="false" --name="exp_cnn_10_0.5_f" > ../logs/exp_cnn_10_0.5_f.txt
# python main.py --algo="FedAvg" --K=100 --C=1 --E=5 --B=10 --T=500 --lr=0.01 --gpu="gpu" --model="cnn" --iid="false" --name="exp_cnn_10_1_f" > ../logs/exp_cnn_10_1_f.txt


# Figure 2
#iid cnn
# for b in 8 50 10
# do
#     for e in 1 5 20 
#     do
#         python3 main.py \
#             --algo="FedAvg" --K=100 --C=0.1 --E=$e --B=$b --T=1000 --lr=0.01 --gpu="gpu" --model="cnn" --name="fig2_cnn_B${b}_E${e}" > ../new_logs/mnist/"fig2_cnn_B${b}_E${e}"
          
#     done     
# done 

# ## Pathological Non-IID split cnn
# for b in 8 50 10
# do
#     for e in 1 5 20 
#     do
#         python3 main.py \
#             --algo="FedAvg" --K=100 --C=0.1 --E=$e --B=$b --T=1000 --lr=0.01 --gpu="gpu" --model="cnn" --iid="false" --name="niid_fig2_cnn_B${b}_E${e}" > ../new_logs/mnist/"niid_fig2_cnn_B${b}_E${e}"
          
#     done     
# done 


# # Figure 7
# #iid 2nn
# for b in 8 50 10
# do
#     for e in 1 10 20 
#     do
#         python3 main.py \
#             --algo="FedAvg" --K=100 --C=0.1 --E=$e --B=$b --T=10 --lr=0.01 --gpu="gpu" --model="nn" --name="fig2_nn_B${b}_E${e}" > ../new_logs/mnist/"fig2_nn_B${b}_E${e}"
          
#     done     
# done 

# ## Pathological Non-IID split 2nn
# for b in 8 50 10
# do
#     for e in 1 10 20 
#     do
#         python3 main.py \
#             --algo="FedAvg" --K=100 --C=0.1 --E=$e --B=$b --T=10 --lr=0.01 --gpu="gpu" --model="nn" --iid="false" --name="niid_fig2_nn_B${b}_E${e}" > ../new_logs/mnist/"niid_fig2_nn_B${b}_E${e}"
          
#     done     
# done 




# #iid resnet
# for b in 8 50 10
# do
#     for e in 1 5 20 
#     do
#         python3 main.py \
#             --algo="FedAvg" --K=100 --C=0.1 --E=$e --B=$b --T=2000 --lr=0.01 --gpu="gpu" --model="resnet" --name="resnet_B${b}_E${e}" > ../new_logs/cifar-100/"resnet_B${b}_E${e}"
          
#     done     
# done 

# ## Pathological Non-IID split resnet
# for b in 8 50 10
# do
#     for e in 1 5 20 
#     do
#         python3 main.py \
#             --algo="FedAvg" --K=100 --C=0.1 --E=$e --B=$b --T=2000 --lr=0.01 --gpu="gpu" --model="resnet" --iid="false" --name="niid_resnet_B${b}_E${e}" > ../new_logs/cifar-100/"niid_resnet_B${b}_E${e}"
          
#     done     
# done 


#-----
#iid resnet
# for b in 8 50 10
# do
#     for e in 1 5 20 
#     do
#         python3 main.py \
#             --algo="FedAvg" --dataset 'cifar-100' --K=100 --C=0.1 --E=$e --B=$b --T=1000 --lr=0.01 --gpu="gpu" --model="resnet" --name="resnet_B${b}_E${e}" > ../new_logs/cifar-100/"resnet_B${b}_E${e}"
          
#     done     
# done 

## Pathological Non-IID split resnet
# for b in 8 50 10
# do
#     for e in 1 5 20   
#     do
#         python3 main.py \
#             --algo="FedAvg" --dataset 'cifar-100' --K=100 --C=0.1 --E=$e --B=$b --T=1000 --lr=0.01 --gpu="gpu" --model="resnet" --iid="false" --name="niid_resnet_B${b}_E${e}" > ../new_logs/cifar-100/"niid_resnet_B${b}_E${e}"
          
#     done     
# done 

# python3 main.py --algo="FedAvg" --dataset 'mnist' --K=100 --C=0.1 --E=1 --B=50 --T=250 --lr=0.1 --gpu="gpu" --model="MNIST_L5" --iid="true" --retrainingR=100 --pretrainingR=25 --name="miid_cnn_E1_T250" > ../new_logs/mnist/"miid_cnn_E1_T250"
# python3 main.py --algo="FedAvg" --dataset 'mnist' --K=100 --C=0.1 --E=5 --B=50 --T=200 --lr=0.1 --gpu="gpu" --model="MNIST_L5" --iid="true" --retrainingR=100 --pretrainingR=25 --name="miid_cnn_E5_T200" > ../new_logs/mnist/"miid_cnn_E5_T200"
# python3 main.py --algo="FedAvg" --dataset 'mnist' --K=100 --C=0.1 --E=10 --B=50 --T=150 --lr=0.1 --gpu="gpu" --model="MNIST_L5" --iid="true" --retrainingR=100 --pretrainingR=25 --name="miid_cnn_E10_T150" > ../new_logs/mnist/"miid_cnn_E10_T150"
# python3 main.py --algo="FedAvg" --dataset 'mnist' --K=100 --C=0.1 --E=20 --B=50 --T=100 --lr=0.1 --gpu="gpu" --model="MNIST_L5" --iid="true" --retrainingR=100 --pretrainingR=25 --name="miid_cnn_E20_T100" > ../new_logs/mnist/"miid_cnn_E20_T100"


# python3 main.py --algo="FedAvg" --dataset 'cifar-10' --K=100 --C=0.1 --E=1 --B=50 --T=250 --lr=0.01 --gpu="gpu" --model="resnet" --iid="true" --retrainingR=100 --pretrainingR=30 --name="ciid_resnet_E1_T250" > ../new_logs/cifar-10/"ciid_resnet_E1_T250"
# python3 main.py --algo="FedAvg" --dataset 'cifar-10' --K=100 --C=0.1 --E=5 --B=50 --T=200 --lr=0.01 --gpu="gpu" --model="resnet" --iid="true" --retrainingR=100 --pretrainingR=30 --name="ciid_resnet_E5_T200" > ../new_logs/cifar-10/"ciid_resnet_E5_T200"
# python3 main.py --algo="FedAvg" --dataset 'cifar-10' --K=100 --C=0.1 --E=10 --B=50 --T=150 --lr=0.01 --gpu="gpu" --model="resnet" --iid="true" --retrainingR=100 --pretrainingR=30 --name="ciid_resnet_E10_T150" > ../new_logs/cifar-10/"ciid_resnet_E10_T150"
# python3 main.py --algo="FedAvg" --dataset 'cifar-10' --K=100 --C=0.1 --E=20 --B=50 --T=100 --lr=0.01 --gpu="gpu" --model="resnet" --iid="true" --retrainingR=100 --pretrainingR=30 --name="ciid_resnet_E20_T100" > ../new_logs/cifar-10/"ciid_resnet_E20_T100"


          

#python3 main.py --algo="FedAvg" --dataset 'mnist' --K=100 --C=0.1 --E=1 --B=50 --T=5 --lr=0.01 --gpu="gpu" --model="MNIST_L5" --retrainingR=5 --dataset="mnist" --iid="true" --pretrainingR=25 --name="miid_cnn_E1_T250" > ../new_logs/mnist/"miid_cnn_E1_T250"
#python3 main.py --algo="FedAvg" --dataset 'cifar-10' --K=100 --C=0.1 --E=1 --B=50 --T=5 --lr=0.01 --gpu="gpu" --model="resnet" --iid="true" --retrainingR=5 --pretrainingR=25 --name="ciid_resnet_E1_T250" > ../new_logs/cifar-100/"ciid_resnet_E1_T250"


# python3 main.py --algo="FedAvg" --dataset 'mnist' --K=100 --C=0.1 --E=5 --B=100 --T=10 --lr=0.1 --gpu="gpu" --model="cnn" --retrainingR=100 --dataset="mnist" --iid="true" --pretrainingR=25 --num_attackers=0 --finetune=0 --prune=0 --name="exp_mnist_a0_f0_p0"

# python3 main.py --algo="FedAvg" --dataset 'mnist' --K=100 --C=0.1 --E=5 --B=100 --T=10 --lr=0.1 --gpu="gpu" --model="cnn" --retrainingR=100 --dataset="mnist" --iid="true" --pretrainingR=25 --num_attackers=20 --finetune=50 --prune=0 --name="exp_mnist_a20_f50_p0"
# python3 main.py --algo="FedAvg" --dataset 'mnist' --K=100 --C=0.1 --E=5 --B=100 --T=10 --lr=0.1 --gpu="gpu" --model="cnn" --retrainingR=100 --dataset="mnist" --iid="true" --pretrainingR=25 --num_attackers=80 --finetune=50 --prune=0 --name="exp_mnist_a80_f50_p0"

# python3 main.py --algo="FedAvg" --dataset 'mnist' --K=100 --C=0.1 --E=5 --B=100 --T=10 --lr=0.1 --gpu="gpu" --model="cnn" --retrainingR=100 --dataset="mnist" --iid="true" --pretrainingR=25 --num_attackers=20 --finetune=0 --prune=0.7 --name="exp_mnist_a20_f0_p70"
# python3 main.py --algo="FedAvg" --dataset 'mnist' --K=100 --C=0.1 --E=5 --B=100 --T=10 --lr=0.1 --gpu="gpu" --model="cnn" --retrainingR=100 --dataset="mnist" --iid="true" --pretrainingR=25 --num_attackers=80 --finetune=0 --prune=0.7 --name="exp_mnist_a80_f0_p70"

# python3 main.py --algo="FedAvg" --dataset 'mnist' --K=100 --C=0.1 --E=5 --B=100 --T=10 --lr=0.1 --gpu="gpu" --model="cnn" --retrainingR=100 --dataset="mnist" --iid="true" --pretrainingR=25 --num_attackers=80 --finetune=50 --prune=0.7 --name="exp_mnist_a80_f50_p70"



# python3 main.py --algo="FedAvg" --dataset 'cifar-10' --K=100 --C=0.1 --E=10 --B=100 --T=30 --lr=0.01 --gpu="gpu" --model="resnet" --retrainingR=100 --dataset="cifar-10" --iid="true" --pretrainingR=30 --num_attackers=0 --finetune=0 --prune=0 --name="exp_cifart_a0_f0_p0"

# python3 main.py --algo="FedAvg" --dataset 'cifar-10' --K=100 --C=0.1 --E=10 --B=100 --T=30 --lr=0.01 --gpu="gpu" --model="resnet" --retrainingR=100 --dataset="cifar-10" --iid="true" --pretrainingR=30 --num_attackers=20 --finetune=50 --prune=0 --name="exp_cifart_a20_f50_p0"
# python3 main.py --algo="FedAvg" --dataset 'cifar-10' --K=100 --C=0.1 --E=10 --B=100 --T=30 --lr=0.01 --gpu="gpu" --model="resnet" --retrainingR=100 --dataset="cifar-10" --iid="true" --pretrainingR=30 --num_attackers=80 --finetune=50 --prune=0 --name="exp_cifart_a80_f50_p0"

# python3 main.py --algo="FedAvg" --dataset 'cifar-10' --K=100 --C=0.1 --E=10 --B=100 --T=30 --lr=0.01 --gpu="gpu" --model="resnet" --retrainingR=100 --dataset="cifar-10" --iid="true" --pretrainingR=30 --num_attackers=20 --finetune=0 --prune=0.7 --name="exp_cifart_a20_f0_p70"
# python3 main.py --algo="FedAvg" --dataset 'cifar-10' --K=100 --C=0.1 --E=10 --B=100 --T=30 --lr=0.01 --gpu="gpu" --model="resnet" --retrainingR=100 --dataset="cifar-10" --iid="true" --pretrainingR=30 --num_attackers=80 --finetune=0 --prune=0.7 --name="exp_cifart_a80_f0_p70"

# python3 main.py --algo="FedAvg" --dataset 'cifar-10' --K=100 --C=0.1 --E=10 --B=100 --T=30 --lr=0.01 --gpu="gpu" --model="resnet" --retrainingR=100 --dataset="cifar-10" --iid="true" --pretrainingR=30 --num_attackers=80 --finetune=50 --prune=0.7 --name="exp_cifart_a80_f50_p70"


#python3 main.py --algo="FedAvg" --dataset 'cifar-100' --K=100 --C=0.1 --E=10 --B=100 --T=30 --lr=0.01 --gpu="gpu" --model="resnet" --retrainingR=100 --dataset="cifar-100" --iid="true" --pretrainingR=30 --num_attackers=0 --finetune=0 --prune=0 --name="exp_cih_a0_f0_p0"



python3 main.py --algo="FedAvg" --dataset 'cifar-100' --K=100 --C=0.1 --E=10 --B=100 --T=50 --lr=0.01 --gpu="gpu" --model="resnet" --retrainingR=100 --dataset="cifar-100" --iid="true" --pretrainingR=25 --num_attackers=0 --finetune=0 --prune=0 --name="exp_cifar100_a0_f0_p0"
python3 main.py --algo="FedAvg" --dataset 'cifar-100' --K=100 --C=0.1 --E=10 --B=100 --T=50 --lr=0.01 --gpu="gpu" --model="resnet" --retrainingR=100 --dataset="cifar-100" --iid="true" --pretrainingR=25 --num_attackers=80 --finetune=0 --prune=0.7 --name="exp_cifar100_a80_f0_p70"
python3 main.py --algo="FedAvg" --dataset 'cifar-100' --K=100 --C=0.1 --E=10 --B=100 --T=50 --lr=0.01 --gpu="gpu" --model="resnet" --retrainingR=100 --dataset="cifar-100" --iid="true" --pretrainingR=25 --num_attackers=80 --finetune=50 --prune=0.7 --name="exp_cifar100_a80_f50_p70"



# python3 main.py --algo="FedAvg" --dataset 'cifar-10' --K=100 --C=0.1 --E=10 --B=100 --T=30 --lr=0.01 --gpu="gpu" --model="resnet" --retrainingR=100 --dataset="cifar-10" --iid="true" --pretrainingR=30 --num_attackers=0 --finetune=0 --prune=0 --name="exp_cifart_a0_f0_p0"

# python3 main.py --algo="FedAvg" --dataset 'cifar-10' --K=100 --C=0.1 --E=10 --B=100 --T=30 --lr=0.01 --gpu="gpu" --model="resnet" --retrainingR=100 --dataset="cifar-10" --iid="true" --pretrainingR=30 --num_attackers=20 --finetune=50 --prune=0 --name="exp_cifart_a20_f50_p0"
# python3 main.py --algo="FedAvg" --dataset 'cifar-10' --K=100 --C=0.1 --E=10 --B=100 --T=30 --lr=0.01 --gpu="gpu" --model="resnet" --retrainingR=100 --dataset="cifar-10" --iid="true" --pretrainingR=30 --num_attackers=80 --finetune=50 --prune=0 --name="exp_cifart_a80_f50_p0"

# python3 main.py --algo="FedAvg" --dataset 'cifar-10' --K=100 --C=0.1 --E=10 --B=100 --T=30 --lr=0.01 --gpu="gpu" --model="resnet" --retrainingR=100 --dataset="cifar-10" --iid="true" --pretrainingR=30 --num_attackers=20 --finetune=0 --prune=0.7 --name="exp_cifart_a20_f0_p70"
# python3 main.py --algo="FedAvg" --dataset 'cifar-10' --K=100 --C=0.1 --E=10 --B=100 --T=30 --lr=0.01 --gpu="gpu" --model="resnet" --retrainingR=100 --dataset="cifar-10" --iid="true" --pretrainingR=30 --num_attackers=80 --finetune=0 --prune=0.7 --name="exp_cifart_a80_f0_p70"

# python3 main.py --algo="FedAvg" --dataset 'cifar-10' --K=100 --C=0.1 --E=10 --B=100 --T=30 --lr=0.01 --gpu="gpu" --model="resnet" --retrainingR=100 --dataset="cifar-10" --iid="true" --pretrainingR=30 --num_attackers=80 --finetune=50 --prune=0.7 --name="exp_cifart_a80_f50_p70"
