#%%
import torch
import numpy as np
import time
from scipy import sparse

#Pytorch dataset
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# PyTorch model
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

class PCBDataset(Dataset):
    def __init__(self, inputs_dataset, outputs_dataset, scalar_dataset):

        assert len(inputs_dataset) == len(outputs_dataset) == len(scalar_dataset), "All datasets must be of the same size"
        self.inputs_dataset = inputs_dataset
        self.outputs_dataset = outputs_dataset
        self.scalar_dataset = scalar_dataset

    def __len__(self):

        return len(self.inputs_dataset)

    def __getitem__(self, idx):

        input_data = self.inputs_dataset[idx]
        output_data = self.outputs_dataset[idx]
        scalar_data = self.scalar_dataset[idx]
        return input_data, output_data, scalar_data
    

class StandardScaler3D:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, tensor):
        self.mean = tensor.mean(dim=[0, 2, 3], keepdim=True)
        self.std = tensor.std(dim=[0, 2, 3], keepdim=True)

    def transform(self, tensor):
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardScaler3D no ha sido ajustado con datos; por favor, llama a .fit() primero.")
        return (tensor - self.mean) / self.std

    def fit_transform(self, tensor):
        self.fit(tensor)
        return self.transform(tensor)
    def inverse_transform(self, tensor):
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardScaler3D no ha sido ajustado con datos; por favor, llama a .fit() primero.")
        mean = self.mean.to(tensor.device)
        std = self.std.to(tensor.device)
        return tensor * std + mean


class GlobalStandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, tensor):
        self.mean = tensor.mean()
        self.std = tensor.std()

    def transform(self, tensor):
        if self.mean is None or self.std is None:
            raise RuntimeError("GlobalStandardScaler no ha sido ajustado con datos; por favor, llama a .fit() primero.")
        return (tensor - self.mean) / (self.std + 1e-6)  # Se añade un pequeño valor para evitar la división por cero

    def fit_transform(self, tensor):
        self.fit(tensor)
        return self.transform(tensor)

    def inverse_transform(self, tensor):
        if self.mean is None or self.std is None:
            raise RuntimeError("GlobalStandardScaler no ha sido ajustado con datos; por favor, llama a .fit() primero.")
        mean = self.mean.to(tensor.device)
        std = self.std.to(tensor.device)
        return tensor * std + mean

class LaEnergiaNoAparece(nn.Module):
    def __init__(self, L:float=0.1,thickness:float=0.001,board_k:float=10,ir_emmisivity:float=0.8):
        super(LaEnergiaNoAparece, self).__init__()
        
        nx = 13
        ny = 13

        self.n_nodes = nx*ny # número total de nodos
        
        interfaces = [0,nx-1,nx*nx-1,nx*nx-nx]

        self.Boltzmann_cte = 5.67E-8

        # cálculo de los GLs y GRs
        dx = L/(nx-1)
        dy = L/(ny-1)
        GLx = thickness*board_k*dy/dx
        GLy = thickness*board_k*dx/dy
        GR = 2*dx*dy*ir_emmisivity

        # Generación de la matriz de acoplamientos conductivos [K]. 
        K_cols = []
        K_rows = []
        K_data = []
        for j in range(ny):
            for i in range(nx):
                id = i + nx*j
                if id in interfaces:
                    K_rows.append(id)
                    K_cols.append(id)
                    K_data.append(1)
                else:
                    GLii = 0
                    if i+1 < nx:
                        K_rows.append(id)
                        K_cols.append(id+1)
                        K_data.append(-GLx)
                        GLii += GLx
                    if i-1 >= 0:
                        K_rows.append(id)
                        K_cols.append(id-1)
                        K_data.append(-GLx)
                        GLii += GLx
                    if j+1 < ny:
                        K_rows.append(id)
                        K_cols.append(id+nx)
                        K_data.append(-GLx)
                        GLii += GLy
                    if j-1 >= 0:
                        K_rows.append(id)
                        K_cols.append(id-nx)
                        K_data.append(-GLx)
                        GLii += GLy
                    K_rows.append(id)
                    K_cols.append(id)
                    K_data.append(GLii)
        indices = torch.LongTensor([K_rows, K_cols])
        values = torch.FloatTensor(K_data)
        shape = torch.Size([self.n_nodes, self.n_nodes])
        self.K = torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float)
        self.K = self.K.cuda()

        E_data = []
        E_id = []
        for id in range(self.n_nodes):
            if id not in interfaces:
                E_id.append(id)
                E_data.append(GR)
        indices = torch.tensor([E_id, E_id], dtype=torch.int64) 
        values = torch.tensor(E_data, dtype=torch.float32)  
        size = torch.Size([self.n_nodes, self.n_nodes])  
        self.E = torch.sparse_coo_tensor(indices, values, size)
        self.E = self.E.cuda()

        self.energyScaler = GlobalStandardScaler()

    def forward(self, outputs, heaters, interfaces, Tenv):
        
        #Generación del vector Q
        heaters = torch.flatten(heaters, start_dim=1)

        interfaces = torch.flatten(interfaces, start_dim=1)


        Q = torch.zeros((outputs.size(0),self.n_nodes),dtype=torch.double)
        for i in range(Q.size(0)):
            for id in range(self.n_nodes):
                if heaters[i,id] != 0:
                    Q[i,id] = heaters[i,id]
                elif interfaces[i,id] != 0:
                    Q[i,id] = interfaces[i,id]

        #Generación del vector T
        T = torch.flatten(outputs, start_dim=1)

        excessEnergy = torch.zeros((Q.size(0),self.n_nodes))


        T, Q, Tenv = T.cuda(), Q.cuda(), Tenv.cuda()


        for i in range(Q.size(0)):
            T_unsqueezed = T[i].unsqueeze(1)
            Tenv_unsqueezed = Tenv[i].unsqueeze(1)
            excessEnergy[i,:] = torch.flatten(torch.sparse.mm(self.K,T_unsqueezed) + self.Boltzmann_cte*torch.sparse.mm(self.E,(T_unsqueezed**4-Tenv_unsqueezed**4)) - Q[i].unsqueeze(1))


        return torch.mean(torch.abs(excessEnergy))

#%%
  
##############################################
############# CARGANDO LOS DATOS #############
##############################################
    

dataset = torch.load('PCB_dataset.pth')

#Estandarizar los datos
scaler_input = StandardScaler3D()
scaler_scalar = GlobalStandardScaler()
scaler_output = GlobalStandardScaler()

scaled_input = scaler_input.fit_transform(dataset.inputs_dataset)
scaled_scalar = scaler_scalar.fit_transform(dataset.scalar_dataset)
scaled_output = scaler_output.fit_transform(dataset.outputs_dataset)

dataset = PCBDataset(scaled_input, scaled_output, scaled_scalar)


diccionario = {
    125:7,
    160:10,
    175:10,
    190:8
}

for train_cases, batch_size in diccionario.items():
    for parameter in [0.25,0.5,0.75,1,1.5]:
        print(f"Train cases: {train_cases}, Batch size: {batch_size}")
        print(f"Parameter: {parameter}")
        # Separando Train and Test
        #train_cases = 60
        test_cases = 1000

        #batch_size = 4
        test_size = 0.1

        #CAMBIAR A GUSTO
        num_train = test_cases + train_cases 
        split = int(np.floor(test_cases))

        #num_train = int(len(dataset)) 
        #split = int(np.floor(test_size * num_train))


        indices = list(range(num_train))

        #SOLO ACTIVAR PARA ESTUDIOS
        seed = 50
        torch.manual_seed(seed) 
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        np.random.shuffle(indices)
        train_idx, test_idx = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        #Creando los Datalaoders
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)



        
        ###################################################
        ############# ARQUITECTURA DEL MODELO #############
        ###################################################

        class TripleConv(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                self.triple_conv = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )

            def forward(self, x):
                return self.triple_conv(x)

        class Down(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                self.maxpool_conv = nn.Sequential(
                    nn.MaxPool2d(2,2),
                    TripleConv(in_channels, out_channels)
                )

            def forward(self, x):
                return self.maxpool_conv(x)

        class SpecialDown(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                self.maxpool_conv = nn.Sequential(
                    nn.AdaptiveMaxPool2d((8,8)),
                    TripleConv(in_channels, out_channels)
                )

            def forward(self, x):
                return self.maxpool_conv(x)

        class Up(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                self.conv = TripleConv(in_channels + in_channels // 2, out_channels)


            def forward(self, x1, x2):
                x1 = self.up(x1)
                x = torch.cat([x2, x1], dim=1)
                return self.conv(x)

        class SpecialUp(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                self.up = nn.Upsample(size=(13,13), mode='bilinear', align_corners=False)
                self.conv = TripleConv(in_channels + in_channels // 2, out_channels)


            def forward(self, x1, x2):
                x1 = self.up(x1)
                x = torch.cat([x2, x1], dim=1)
                return self.conv(x)
            

        class UNet(nn.Module):
            def __init__(self):
                super(UNet, self).__init__()
                
                #Encoder
                self.inc = TripleConv(2, 64)
                self.down1 = SpecialDown(64, 128)
                self.down2 = Down(128, 256)
                self.down3 = Down(256, 512)

                #Scalar
                self.fc_scalar = nn.Linear(1, 32)

                #MLP
                self.fc1 = nn.Linear(2 * 2 * 512 + 32, 2048)
                self.fc2 = nn.Linear(2048,2048)
                self.fc3 = nn.Linear(2048,2048)
                self.fc4 = nn.Linear(2048,2 * 2 * 512)

                #Decoder
                self.up1 = Up(512, 256)
                self.up2 = Up(256, 128)
                self.up3 = SpecialUp(128, 64)
                self.outc = nn.Conv2d(64, 1, kernel_size=1)

                self.dropout = nn.Dropout(p=0)

            def forward(self, x, scalar):

                #Encoder
                x1 = self.inc(x)
                x2 = self.down1(x1)
                x3 = self.down2(x2)
                x4 = self.down3(x3)
                x = torch.flatten(x4, 1)

                #Tenv
                scalar = F.relu(self.fc_scalar(scalar))

                #MLP conjunto
                x = torch.cat((x,scalar), dim=1)
                x = F.relu(self.fc1(x))
                x = self.dropout(x)
                x = F.relu(self.fc2(x))
                x = self.dropout(x)
                x = F.relu(self.fc3(x))
                x = self.dropout(x)
                x = F.relu(self.fc4(x))

                #Decoder
                x = x.view(x.size(0), 512, 2, 2)
                x = self.up1(x, x3)
                x = self.up2(x, x2)
                x = self.up3(x, x1)
                x = self.outc(x)
                return x
            
        model = UNet()
        model.cuda()
        criterion = nn.MSELoss()
        criterionPhysics = LaEnergiaNoAparece()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=15)
        last_test_loss = np.inf

        
        ######################################
        ############# TRAIN LOOP #############
        ######################################

        num_epochs = 100

        start_time = time.time()

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            num_batches = len(train_loader)
            for batch, target, scalar in train_loader: 

                optimizer.zero_grad()

                scalar = scalar.view(scalar.size(0),1)
                batch, scalar, target = batch.cuda(), scalar.cuda(), target.cuda()

                # Forward pass
                outputs = model.forward(batch, scalar)
                outputs = outputs.view(outputs.size(0),13,13)

                #Añadir criterios de fallo
                T_interfaces = torch.zeros((target.size(0), 2, 2))
                T_interfaces[:,0,0], T_interfaces[:,0,1], T_interfaces[:,1,0], T_interfaces[:,1,1] = target[:,0,0], target[:,0,12], target[:,12,0], target[:,12,12]

                outputs_interfaces = torch.zeros((outputs.size(0),2,2))
                outputs_interfaces[:,0,0], outputs_interfaces[:,0,1], outputs_interfaces[:,1,0], outputs_interfaces[:,1,1] = outputs[:,0,0], outputs[:,0,12], outputs[:,12,0], outputs[:,12,12]


                #Desestandarizar para la física
                batch_p = scaler_input.inverse_transform(batch)
                outputs_p = scaler_output.inverse_transform(outputs)
                scalar_p = scaler_scalar.inverse_transform(scalar)

                loss_p = criterionPhysics(outputs_p.view(outputs.size(0),13,13),batch_p[:,0,:,:].view(batch.size(0),13,13),batch_p[:,1,:,:].view(batch.size(0),13,13),scalar_p.view(batch.size(0),1))
                loss = 2*criterion(outputs, target) +  criterion(outputs_interfaces,T_interfaces) + parameter*loss_p

                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                #print(loss_p.item())

            avg_loss = total_loss/num_batches


            scheduler.step(avg_loss)


            last_lr = scheduler.optimizer.param_groups[0]['lr']

            total_loss = 0.0
            total_batches = 0

            model.eval()
            with torch.no_grad(): 
                for batch, target, scalar in test_loader:

                    # Prepare the data and target
                    scalar = scalar.view(scalar.size(0), 1)
                    batch, scalar, target = batch.cuda(), scalar.cuda(), target.cuda()

                    # Forward pass

                    outputs = model(batch, scalar)
                    outputs = outputs.view(outputs.size(0),13,13)

                    loss = criterion(outputs, target)

                    # Accumulate the loss
                    total_loss += loss.item()
                    total_batches += 1

            # Compute the average loss over all batches
            avg_test_loss = total_loss / total_batches

            #print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.8f}, Current LR: {last_lr}")
            if avg_test_loss < last_test_loss:
                #print("{:.8f} -----> {:.8f}   Saving...".format(last_test_loss,avg_test_loss))
                torch.save(model.state_dict(), 'modelos\modelo_PI.pth')
                last_test_loss = avg_test_loss

        end_time = time.time() 
        total_time = end_time - start_time 

        minutes, seconds = divmod(total_time, 60)
        #print(f"Total training time: {int(minutes)} minutes and {int(seconds)} seconds") 


        
        ######################################
        ############# LOAD MODEL #############
        ######################################

        model.load_state_dict(torch.load('modelos\modelo_PI.pth'))
        model.eval()

        # Variables to track losses and the number of batches
        total_loss = 0.0
        total_batches = 0

        with torch.no_grad(): 
            for batch, target, scalar in test_loader:

                # Prepare the data and target
                scalar = scalar.view(scalar.size(0), 1)
                batch, scalar, target = batch.cuda(), scalar.cuda(), target.cuda()

                # Forward pass
                outputs = model(batch, scalar)
                outputs = outputs.view(outputs.size(0),13,13)

                loss = criterion(outputs, target)

                # Accumulate the loss
                total_loss += loss.item()
                total_batches += 1

        # Compute the average loss over all batches
        avg_test_loss = total_loss / total_batches
        #print(f'Test Loss: {avg_test_loss:.6f}')
        last_test_loss = avg_test_loss

        
        #####################################
        ############# TEST LOOP #############
        #####################################

        # Ensure the model is in evaluation mode
        model.eval()

        # Variables to track losses and the number of batches
        total_loss = 0.0
        total_batches = 0

        with torch.no_grad(): 
            for batch, target, scalar in test_loader:

                # Prepare the data and target
                scalar = scalar.view(scalar.size(0), 1)
                batch, scalar, target = batch.cuda(), scalar.cuda(), target.cuda()

                # Forward pass
                outputs = model(batch, scalar)
                outputs = outputs.view(outputs.size(0),13,13)

                outputs = scaler_output.inverse_transform(outputs)
                target = scaler_output.inverse_transform(target)

                loss = criterion(outputs, target)

                # Accumulate the loss
                total_loss += loss.item()
                total_batches += 1

        # Compute the average loss over all batches
        avg_test_loss = total_loss / total_batches
        print(f'Test Loss: {avg_test_loss:.6f}')


# %%
##############################################
############# MOSTRAR RESULTADOS #############
##############################################

import matplotlib.pyplot as plt

plt.style.use('default')

plt.rcParams["figure.figsize"] = (6,4)

#plt.rcParams["font.family"] = "Times New Roman"

plt.rcParams["font.family"] = "lmroman10-regular"

plt.rcParams["font.size"] = 12

plt.rcParams["text.usetex"] = True

plt.rcParams["axes.titlesize"] = 11

model.eval()
# Función para visualizar el output de la red y el target
def visualizar_valores_pixeles(output, target):
    # Convertir los tensores a numpy y asegurarse de que están en CPU
    output_np = output.squeeze().cpu().detach().numpy()
    target_np = target.squeeze().cpu().detach().numpy()
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    # Asegurarse de que las celdas de la grilla sean lo suficientemente grandes para el texto
    fig.tight_layout(pad=3.0)
    
    # Output de la red
    axs[1].imshow(output_np, cmap='viridis', interpolation='nearest')
    axs[1].title.set_text('Output')
    for i in range(output_np.shape[0]):
        for j in range(output_np.shape[1]):
            text = axs[1].text(j, i, f'{output_np[i, j]:.0f}',
                               ha="center", va="center", color="w", fontsize=6)

    # Target
    axs[0].imshow(target_np, cmap='viridis', interpolation='nearest')
    axs[0].title.set_text('Target')
    for i in range(target_np.shape[0]):
        for j in range(target_np.shape[1]):
            text = axs[0].text(j, i, f'{target_np[i, j]:.0f}',
                               ha="center", va="center", color="w", fontsize=6)

    plt.savefig('PlantillaV.png', dpi=300, bbox_inches='tight')

    plt.show()

count = 0 
with torch.no_grad(): 
    for batch, target, scalar in test_loader:

        # Prepare the data and target
        scalar = scalar.view(scalar.size(0), 1)
        batch, scalar, target = batch.cuda(), scalar.cuda(), target.cuda()

        # Forward pass
        outputs = model(batch, scalar)
        

        outputs = scaler_output.inverse_transform(outputs)
        target = scaler_output.inverse_transform(target)
        for i in range(5):
            visualizar_valores_pixeles(outputs[i], target[i])
            count += 1
        if count>= 5: break
        



# %%
