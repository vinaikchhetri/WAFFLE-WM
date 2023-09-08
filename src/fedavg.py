from functools import reduce
import torch
from torch.utils.data import Dataset
import utils
import numpy as np
import models
import torch.optim as optim
from utils import accuracy
from utils import moving_average

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
		#return torch.tensor(img), torch.tensor(label)
		return img, label

#FedAvg Algorithm
def fed_avg(args, client_data_dict, trainset, testset): #args, dictionary of clients' data.
	if args.gpu == "gpu":
		device = torch.device('cuda:0')
	else:
		device = torch.device('cpu')
	
	if args.model == 'nn':
		model_global = models.MP(28*28,200,10)
		model_global.to(device)
		model_local = models.MP(28*28,200,10)
		criterion = torch.nn.CrossEntropyLoss()
	
	if args.model == 'cnn':
		model_global = models.CNN_MNIST()
		model_global.to(device)
		model_local = models.CNN_MNIST()
		criterion = torch.nn.CrossEntropyLoss()
		
	w_global = model_global.state_dict()
	num_samples_dict = {} #number of samples per user.

	for i in range (args.K): #loop through clients
		num_samples_dict[i] = len(client_data_dict[i]) 
	best_test_acc = -1
	for rounds in range(args.T): #total number of rounds
		m = int(max(args.C*args.K, 1)) 
		client_indices = np.random.choice(args.K, m, replace=False)
		#client_indices.astype(int)
		num_samples_list = [num_samples_dict[idx] for idx in client_indices] #list containing number of samples for each user id.
		total_num_samples = reduce(lambda x,y: x+y, num_samples_list, 0)
		store = {}
		for index,client_idx in enumerate(client_indices): #loop through selected clients
			#model_local = models.MP(28*28,200,10)
			#model_local.to(device)
			w_local = client_update(trainset, client_idx, w_global.copy(), args, client_data_dict, criterion, model_local) #client index, global weight, args, dictionary of clients' data, criterion, optimizer.
			store[index] = w_local
		##Moving Average
		# 	if index==0: #moving average does not exist when the first client is selected.
		# 		moving_weights = w_local
		# 		for layer in w_local:
		# 			moving_weights[layer] = w_local[layer]*(num_samples_list[index]/total_num_samples)
		# 	else:
		# 		moving_weights = moving_average(moving_weights, w_local, num_samples_list[index], total_num_samples)
		# 		#temp = w_local * (num_samples_list[index]/total_num_samples)
		# w_global =  moving_weights

		w_global = {}
		for layer in store[0]:
			sum = 0
			for user_key in store:
				sum += store[user_key][layer]*num_samples_list[user_key]/total_num_samples
			w_global[layer] = sum

		#Performing evaluation on test data.
		model_global.load_state_dict(w_global)
		test_loader = torch.utils.data.DataLoader(testset, batch_size=64,
											shuffle=False)
		with torch.no_grad():
			model_global.eval()
			running_loss = 0.0
			running_acc = 0.0
			for index,data in enumerate(test_loader):  
				inputs, labels = data
				inputs = inputs.to(device)
				if args.model == 'nn':
					inputs = inputs.flatten(1)
				labels = labels.to(device)
				output = model_global(inputs)
				loss = criterion(output, labels)
				pred = torch.argmax(output, dim=1)
				acc = accuracy(pred, labels)
				running_acc += acc
				running_loss += loss
		avg_acc = running_acc / (index+1)
		if best_test_acc<avg_acc:
			best_test_acc = avg_acc
			best_dict = {rounds: best_test_acc}
			#torch.save(best_dict,'nn-trial.pt')
		print('Round '+ str(rounds))
		print(f'server stats: [loss: {running_loss / (index+1):.3f}')
		print(f'server stats: [accuracy: {running_acc / (index+1):.3f}')

def client_update(trainset, client_idx, w_global, args, client_data_dict, criterion, model_local):
	model_local.load_state_dict(w_global)

	data_client = client_data_dict[client_idx] #client i's data.
	
	cd = CustomDataset(trainset, data_client)
	data_loader = torch.utils.data.DataLoader(cd, batch_size=args.B,
                                                shuffle=True)
	if args.gpu == "gpu":
		device = torch.device('cuda:0')
	else:
		device = torch.device('cpu')
	
	model_local.to(device)
	optimizer = optim.SGD(model_local.parameters(), lr=0.01, momentum=0.5)

	for epoch in range(args.E):
		running_loss = 0.0
		running_acc = 0.0
		for index,data in enumerate(data_loader):        
			inputs, labels = data
			#print("inputs ", inputs.shape)
			#print("labels ", labels)
			inputs = inputs.to(device)
			labels = labels.to(device)

		
			optimizer.zero_grad()
			if args.model == 'nn':
				inputs = inputs.flatten(1)
			outputs = model_local(inputs)
			pred = torch.argmax(outputs, dim=1)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			# print statistics
			acc = accuracy(pred, labels)
			running_acc += acc
			running_loss += loss.item()
			if index % 100 == 99:    # print every 2000 mini-batches
				#print(f'device: {client_idx} [{epoch + 1}, {index + 1:5d}] loss: {running_loss / 100:.3f}')
				#print(f'device: {client_idx} [{epoch + 1}, {index + 1:5d}] accuracy: {running_acc / 100:.3f}')
				running_loss = 0.0
				running_acc = 0.0

	# print('Finished Training Device '+ str(client_idx))
	return model_local.state_dict()


