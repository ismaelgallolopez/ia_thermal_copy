#%%
import torch
import numpy as np

#Pytorch dataset
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# PyTorch model
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class PCBDataset(Dataset):
    def __init__(self, inputs_tensor, outputs_tensor):
        self.inputs = inputs_tensor
        self.outputs = outputs_tensor

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Directly return the slices of the tensors
        return self.inputs[idx], self.outputs[idx]
    
class VectorDataset(Dataset):
    def __init__(self, vector):
        self.vector = vector

    def __len__(self):
        return len(self.vector)

    def __getitem__(self, idx):
        return self.vector[idx]
#%%
##############################################
############# CARGANDO LOS DATOS #############
##############################################
    
dataset = torch.load('PCB_dataset.pth')
scalar_dataset = torch.load('PCB_dataset_scalar.pth')

# Separando Train and Test
test_size = 0.1
num_train = len(dataset)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(test_size * num_train))
train_idx, test_idx = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_idx)
test_sampler = SubsetRandomSampler(test_idx)

#Creando los Datalaoders
train_loader = DataLoader(dataset, batch_size=64, sampler=train_sampler)
train_scalar_loader = DataLoader(scalar_dataset, batch_size=64, sampler=train_sampler)
test_loader = DataLoader(dataset, batch_size=64, sampler=test_sampler)
test_scalar_loader = DataLoader(scalar_dataset, batch_size=64, sampler=test_sampler)

#%%
###################################################
############# ARQUITECTURA DEL MODELO #############
###################################################

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        #Concolutional
        self.conv1 = nn.Conv2d(2, 10, 3, padding=1)
        self.conv2 = nn.Conv2d(10, 20, 3, padding=1)
        self.conv3 = nn.Conv2d(20, 40, 3, padding=1)

        #Pooling
        self.adaptative_pool = nn.AdaptiveMaxPool2d((8,8))
        self.pool = nn.MaxPool2d(2,2)

        #Scalar
        self.fc_scalar = nn.Linear(1, 32)

        #MLP
        self.fc1 = nn.Linear(2 * 2 * 40 + 32, 80)
        self.fc2 = nn.Linear(80,40)
        self.fc3 = nn.Linear(40,4)

        self.lrelu = nn.LeakyReLU()

        self.dropout = nn.Dropout(p=0.3)


    def forward(self, x, scalar):

        #Parte convolucional
        x = F.relu(self.conv1(x))
        x = self.adaptative_pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)

        #Parte Tenv
        scalar = F.relu(self.fc_scalar(scalar))

        #Parte MLP conjunto
        x = torch.cat((x,scalar), dim=1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))

        return x
    
model = Net()
model.cuda()

#%%
######################################
############# TRAIN LOOP #############
######################################

batch_iterator = iter(train_loader)
scalar_iterator = iter(train_scalar_loader)

num_epochs = 600

criterion = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=0.001/len(train_loader))

for epoch in range(num_epochs):
    model.train()
    for _ in range(len(train_loader)): 
        try:
            batch, target = next(batch_iterator)
            scalar = next(scalar_iterator)
        except StopIteration:
            # Reinitialize the iterators for the next epoch
            batch_iterator = iter(train_loader)
            scalar_iterator = iter(train_scalar_loader)
            batch, target = next(batch_iterator)
            scalar = next(scalar_iterator)

        
        optimizer.zero_grad()
        real_target = torch.zeros((target.size(0),2,2))
        real_target[:,0,0], real_target[:,1,1], real_target[:,1,0], real_target[:,0,1] = target[:,6,3], target[:,6,9], target[:,3,6], target[:,9,6]
        scalar = scalar.view(scalar.size(0),1)
        batch, scalar, real_target = batch.cuda(), scalar.cuda(), real_target.cuda()
        # Forward pass
        outputs = model.forward(batch, scalar)
        real_target = torch.flatten(real_target, 1)
        loss = criterion(outputs, real_target)
        #print(outputs)
        #print(real_target)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
# %%
#Guardar el modelo
torch.save(model.state_dict(), 'modelos\modelo_potencias_V1.2.pth')

#%%
#####################################
############# TEST LOOP #############
#####################################

    # Ensure the model is in evaluation mode
model.eval()

# Variables to track losses and the number of batches
total_loss = 0.0
total_batches = 0

with torch.no_grad():  # Disable gradient computation
    # Initialize iterators for the test datasets
    batch_iterator = iter(test_loader)
    scalar_iterator = iter(test_scalar_loader)

    for _ in range(len(test_loader)):
        try:
            batch, target = next(batch_iterator)
            scalar = next(scalar_iterator)
        except StopIteration:
            break  # Shouldn't happen with properly sized datasets

        # Prepare the data and target
        real_target = torch.zeros((target.size(0), 2, 2))
        real_target[:, 0, 0], real_target[:, 1, 1], real_target[:, 1, 0], real_target[:, 0, 1] = target[:, 6, 3], target[:, 6, 9], target[:, 3, 6], target[:, 9, 6]
        scalar = scalar.view(scalar.size(0), 1)
        batch, scalar, real_target = batch.cuda(), scalar.cuda(), real_target.cuda()

        # Forward pass
        outputs = model(batch, scalar)
        real_target = torch.flatten(real_target, 1)
        print(outputs[0])
        print(real_target[0])

        # Compute the loss
        loss = criterion(outputs, real_target)

        # Accumulate the loss
        total_loss += loss.item()
        total_batches += 1

# Compute the average loss over all batches
avg_loss = total_loss / total_batches
print(f'Test Loss: {avg_loss:.4f}')


# %%
import matplotlib.pyplot as plt

# Put the model in evaluation mode
model.eval()

# Keep track of the count to stop after 10 samples
count = 0

# Disable gradient computation
with torch.no_grad():
    for batch, target in test_loader:
        scalar_iterator = iter(test_scalar_loader)  # Prepare to iterate over scalars
        scalar = next(scalar_iterator)

        # Ensure the scalar tensor is properly shaped and move data to the correct device if using CUDA
        scalar = scalar.view(scalar.size(0), 1)
        if torch.cuda.is_available():
            batch, scalar, target = batch.cuda(), scalar.cuda(), target.cuda()

        real_target = torch.zeros((target.size(0), 2, 2))
        real_target[:,0,0], real_target[:,1,1], real_target[:,1,0], real_target[:,0,1] = target[:,6,3], target[:,6,9], target[:,3,6], target[:,9,6]
        # Forward pass to get outputs
        outputs = model(batch, scalar)
        
        # Convert outputs and target to CPU for visualization if necessary
        outputs = outputs.cpu()
        real_target = real_target.cpu()

        for i in range(batch.size(0)):
            if count >= 10:
                break  # Break the loop after visualizing 10 samples
            
            plt.figure(figsize=(6, 3))

            # Plot real target
            plt.subplot(1, 2, 1)
            plt.imshow(real_target[i].view(2, 2), cmap='hot', interpolation='nearest')
            plt.title('Real Target')
            plt.colorbar()

            # Plot model output
            plt.subplot(1, 2, 2)
            plt.imshow(outputs[i].detach().view(2, 2), cmap='hot', interpolation='nearest')
            plt.title('Model Output')
            plt.colorbar()

            plt.show()
            print("Real Target:", real_target[i])
            print("Model Output:", outputs[i])

            count += 1
        if count >= 10:
            break  # Break the loop after visualizing 10 samples

# %%
