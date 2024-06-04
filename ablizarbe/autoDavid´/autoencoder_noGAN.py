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

class PartitionedStandardScaler:
    def __init__(self):
        self.mean_first_part = None
        self.std_first_part = None
        self.mean_second_part = None
        self.std_second_part = None

    def fit(self, data):
        # Ensure data is a tensor with the second dimension being 9
        if data.shape[1] != 9:
            raise ValueError("Each sample must be a 9-item vector.")

        # Split data into first part (first 4 items) and second part (last 5 items)
        first_part = data[:, :4]
        second_part = data[:, 4:]

        # Calculate mean and std for both parts
        self.mean_first_part = first_part.mean(dim=0, keepdim=True)
        self.std_first_part = first_part.std(dim=0, keepdim=True)
        self.mean_second_part = second_part.mean(dim=0, keepdim=True)
        self.std_second_part = second_part.std(dim=0, keepdim=True)

    def transform(self, data):
        if self.mean_first_part is None or self.std_first_part is None or \
           self.mean_second_part is None or self.std_second_part is None:
            raise RuntimeError("Scaler has not been fitted with data; please call .fit() first.")

        # Apply normalization separately
        first_part_transformed = (data[:, :4] - self.mean_first_part) / self.std_first_part
        second_part_transformed = (data[:, 4:] - self.mean_second_part) / self.std_second_part

        # Concatenate the transformed parts back together
        return torch.cat((first_part_transformed, second_part_transformed), dim=1)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data):
        if self.mean_first_part is None or self.std_first_part is None or \
           self.mean_second_part is None or self.std_second_part is None:
            raise RuntimeError("Scaler has not been fitted with data; please call .fit() first.")

        # Apply inverse transformation separately
        first_part_inversed = data[:, :4] * self.std_first_part + self.mean_first_part
        second_part_inversed = data[:, 4:] * self.std_second_part + self.mean_second_part

        # Concatenate the inversely transformed parts back together
        return torch.cat((first_part_inversed, second_part_inversed), dim=1)

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
        
        self.primeravez = True

        nx = 13
        ny = 13

        self.n_nodes = nx*ny # número total de nodos
        
        interfaces = [0,self.n_nodes-1,self.n_nodes*self.n_nodes-1,self.n_nodes*self.n_nodes-self.n_nodes]

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
                    K_data.append(0)
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


        #Generación del vector T
        T = torch.flatten(outputs, start_dim=1)

        excessEnergy = torch.zeros((Q.size(0),self.n_nodes))

        T, Q, Tenv = T.cuda(), Q.cuda(), Tenv.cuda()


        for i in range(Q.size(0)):
            T_unsqueezed = T[i].unsqueeze(1)
            Tenv_unsqueezed = Tenv[i].unsqueeze(1)
            excessEnergy[i,:] = torch.flatten(torch.sparse.mm(self.K,T_unsqueezed) + self.Boltzmann_cte*torch.sparse.mm(self.E,(T_unsqueezed**4-Tenv_unsqueezed**4)) - Q[i].unsqueeze(1))

        if self.primeravez:
            self.energyScaler.fit(excessEnergy)
            self.mean = self.energyScaler.mean.clone().detach().requires_grad_(True)
            self.std = self.energyScaler.std.clone().detach().requires_grad_(True)
            self.primeravez = False
            
        #excessEnergy = (excessEnergy- self.mean)/self.std
        #print(torch.mean(torch.abs(excessEnergy)))

        return torch.mean(torch.abs(excessEnergy))
    
#%%
##############################################
############# CARGANDO LOS DATOS #############
##############################################
dataset = torch.load('PCB_dataset.pth')

#Estandarizar los datos
scaler_input = PartitionedStandardScaler()
scaler_scalar = GlobalStandardScaler()
scaler_output = GlobalStandardScaler()

scaled_input = scaler_input.fit_transform(dataset.inputs_dataset)
scaled_scalar = scaler_scalar.fit_transform(dataset.scalar_dataset)
scaled_output = scaler_output.fit_transform(dataset.outputs_dataset)

dataset = PCBDataset(scaled_input, scaled_output, scaled_scalar)

capasmult = 32
# Separando Train and Test
train_cases = 20
test_cases = 1000

batch_size = 50
test_size = 0.1

#CAMBIAR A GUSTO
#num_train = test_cases + train_cases 
#split = int(np.floor(test_cases))

num_train = int(len(dataset)) 
split = int(np.floor(test_size * num_train))


indices = list(range(num_train))

#SOLO ACTIVAR PARA ESTUDIOS
seed = 50
#torch.manual_seed(seed) 
#torch.cuda.manual_seed(seed)
#np.random.seed(seed)

np.random.shuffle(indices)

train_idx, test_idx = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_idx)
test_sampler = SubsetRandomSampler(test_idx)

#Creando los Datalaoders
train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

#%%
###################################################
############# ARQUITECTURA DEL MODELO #############
###################################################


# Definir el Encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 16*capasmult, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16*capasmult, 32*capasmult, kernel_size=3, padding=1)  
        self.conv3 = nn.Conv2d(32*capasmult, 16*capasmult, kernel_size=3, padding=1)  
        self.conv4 = nn.Conv2d(16*capasmult, 4*capasmult, kernel_size=3, padding=1) 
        self.pool = nn.MaxPool2d(2, 2) 
        self.adaptativepool = nn.AdaptiveAvgPool2d((8,8))
        self.linear1 = nn.Linear(4*2*2*capasmult, 9)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = self.adaptativepool(x)
        x = F.leaky_relu(self.conv2(x))
        x = self.pool(x)
        x = F.leaky_relu(self.conv3(x))
        x = self.pool(x)
        x = F.leaky_relu(self.conv4(x))
        x = x.view(-1, 4*2*2*capasmult)
        x = self.linear1(x)
        return x

# Definir el Decodificador General (GCD)
class GeneralDecoder(nn.Module):
    def __init__(self):
        super(GeneralDecoder, self).__init__()
        self.linear1 = nn.Linear(9, 4*2*2*capasmult)
        self.conv1 = nn.Conv2d(4*capasmult, 16*capasmult, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16*capasmult, 32*capasmult, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32*capasmult, 16*capasmult, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(16*capasmult, 1, kernel_size=3, padding=1)
        self.upsampleSpecial = nn.Upsample(size=(13,13), mode='bilinear', align_corners=True)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.linear1(x)
        x = x.view(-1, 4*capasmult, 2, 2)
        x = F.leaky_relu(self.conv1(x))
        x = self.up(x)
        x = F.leaky_relu(self.conv2(x))
        x = self.up(x)
        x = F.leaky_relu(self.conv3(x))
        x = self.upsampleSpecial(x)
        x = self.conv4(x)
        return x

# Definir el Decodificador Residual (RCD)
class ResidualDecoder(nn.Module):
    def __init__(self):
        super(ResidualDecoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 16*capasmult, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16*capasmult, 32*capasmult, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32*capasmult, 16*capasmult, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(16*capasmult, 1, kernel_size=3, padding=1)

    def forward(self, x):
        input = x
        residual = F.leaky_relu(self.conv1(x))
        residual = F.leaky_relu(self.conv2(residual))
        residual = F.leaky_relu(self.conv3(residual))
        residual = self.conv4(residual)
        return input + residual  # Sumar el input con el residuo para obtener el output

# Definir los modelos
encoder = Encoder()
general_decoder = GeneralDecoder()
residual_decoder = ResidualDecoder()


# Definir los optimizadores
optimizer = optim.Adam(list(encoder.parameters()) + list(general_decoder.parameters()) + list(residual_decoder.parameters()), lr=0.0001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=15)

# Definir las funciones de pérdida
criterionReconstruction = nn.MSELoss()

last_test_loss = np.inf


#%%
######################################
############# TRAIN LOOP #############
######################################

# Número de épocas
num_epochs = 75

# Ruta para guardar los modelos
base_path = 'modelos\modeloPru_{}.pth'

models = {
    "encoder": encoder,
    "generalDecoder": general_decoder,
    "residualDecoder": residual_decoder,
}


# Enviar etiquetas al dispositivo correcto, por ejemplo 'cuda' si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder, general_decoder, residual_decoder = encoder.to(device), general_decoder.to(device), residual_decoder.to(device)

for epoch in range(num_epochs):

    encoder.train()
    general_decoder.train()
    residual_decoder.train()

    total_loss = 0
    num_batches = len(train_loader)

    for _, target, _ in train_loader: 

        target = target.view(-1, 1, 13, 13).to(device)  # Asegurándonos que el tensor está en el dispositivo correcto
        batch_si = target.size(0)
        
        ### Entrenamiento del generador y encoder
        optimizer.zero_grad()
        
        encoded = encoder(target)
        general_decoded = general_decoder(encoded)
        residual_decoded = residual_decoder(general_decoded)
        residual_decoded = residual_decoded.view(-1, 13,13)
        target = target.view(-1, 13, 13)
        
        gen_loss = criterionReconstruction(residual_decoded, target)
        
        g_loss = gen_loss
        g_loss.backward()
        optimizer.step()
        total_loss += g_loss.item()
    
    avg_loss = total_loss/num_batches
    scheduler.step(avg_loss)
    last_lr = scheduler.optimizer.param_groups[0]['lr']
    
    # Poner los modelos en modo de evaluación
    encoder.eval()
    general_decoder.eval()
    residual_decoder.eval()

    # Desactivar el cálculo de gradientes para la evaluación
    with torch.no_grad():
        total_loss = 0
        for _, target, _ in test_loader:
            target = target.view(-1, 1, 13, 13).to(device)  # Asegurándonos que el tensor está en el dispositivo correcto
            
            # Pasar los datos por el modelo
            encoded = encoder(target)
            general_decoded = general_decoder(encoded)
            residual_decoded = residual_decoder(general_decoded)
            residual_decoded = residual_decoded.view(-1, 13,13)
            target = target.view(-1, 13, 13)
            
            # Calcular la pérdida
            loss = criterionReconstruction(residual_decoded, target)
            total_loss += loss.item()
            
        # Calcular la pérdida promedio
        avg_test_loss = total_loss / len(test_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Generator Loss: {avg_loss:.8f}, Current LR: {last_lr}")
        
        if avg_test_loss < last_test_loss:
            print("{:.8f} -----> {:.8f}   Saving...".format(last_test_loss,avg_test_loss))
            for name, model in models.items():
                model_path = base_path.format(name)
                torch.save(model.state_dict() , model_path)
            last_test_loss = avg_test_loss
        


#%%
#######################################
############# LOAD MODEL ##############
#######################################
    
# Cargar los modelos
base_path = 'modelos\modeloPru_{}.pth'

models = {
    "encoder": encoder,
    "generalDecoder": general_decoder,
    "residualDecoder": residual_decoder,
}

for name, model in models.items():
    model_path = base_path.format(name)
    model.load_state_dict(torch.load(model_path))


# Enviar etiquetas al dispositivo correcto, por ejemplo 'cuda' si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder, general_decoder, residual_decoder = encoder.to(device), general_decoder.to(device), residual_decoder.to(device)

# Poner los modelos en modo de evaluación
encoder.eval()
general_decoder.eval()
residual_decoder.eval()

# Desactivar el cálculo de gradientes para la evaluación
with torch.no_grad():
    total_loss = 0
    for _, target, _ in test_loader:
        target = target.view(-1, 1, 13, 13).to(device)  # Asegurándonos que el tensor está en el dispositivo correcto
        
        # Pasar los datos por el modelo
        encoded = encoder(target)
        general_decoded = general_decoder(encoded)
        residual_decoded = residual_decoder(general_decoded)
        residual_decoded = residual_decoded.view(-1, 13,13)
        target = target.view(-1, 13, 13)

        # Calcular la pérdida
        loss = criterionReconstruction(residual_decoded, target)
        total_loss += loss.item()
        
    # Calcular la pérdida promedio
    avg_test_loss = total_loss / len(test_loader)

last_test_loss = avg_test_loss

print("Test Loss: {:.8f}".format(avg_test_loss))

#%%
######################################
############# TEST LOOP ##############
######################################

# Enviar etiquetas al dispositivo correcto, por ejemplo 'cuda' si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder, general_decoder, residual_decoder = encoder.to(device), general_decoder.to(device), residual_decoder.to(device)

    
# Poner los modelos en modo de evaluación
encoder.eval()
general_decoder.eval()
residual_decoder.eval()

# Desactivar el cálculo de gradientes para la evaluación
with torch.no_grad():
    total_loss = 0
    for _, target, _ in test_loader:
        target = target.view(-1, 13*13).to(device)  # Asegurándonos que el tensor está en el dispositivo correcto
        
        # Pasar los datos por el modelo
        encoded = encoder(target)
        general_decoded = general_decoder(encoded)
        residual_decoded = residual_decoder(general_decoded)
        
        # Calcular la pérdida
        loss = criterionReconstruction(residual_decoded, target)
        total_loss += loss.item()
        
    # Calcular la pérdida promedio
    avg_test_loss = total_loss / len(test_loader)

print("Test Loss: {:.8f}".format(avg_test_loss))

#%%
##############################################
############# MOSTRAR RESULTADOS #############
##############################################

import matplotlib.pyplot as plt

# Enviar etiquetas al dispositivo correcto, por ejemplo 'cuda' si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder, general_decoder, residual_decoder = encoder.to(device), general_decoder.to(device), residual_decoder.to(device)


encoder.eval()
general_decoder.eval()
residual_decoder.eval()

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
    axs[1].title.set_text('Output de la Red')
    for i in range(output_np.shape[0]):
        for j in range(output_np.shape[1]):
            text = axs[1].text(j, i, f'{output_np[i, j]:.1f}',
                               ha="center", va="center", color="w", fontsize=6)

    # Target
    axs[0].imshow(target_np, cmap='viridis', interpolation='nearest')
    axs[0].title.set_text('Target')
    for i in range(target_np.shape[0]):
        for j in range(target_np.shape[1]):
            text = axs[0].text(j, i, f'{target_np[i, j]:.1f}',
                               ha="center", va="center", color="w", fontsize=6)

    plt.show()

count = 0 
with torch.no_grad(): 
    for batch, target, scalar in test_loader:

        # Prepare the data and target
        target = target.view(-1, 1, 13,13).to(device)

        # Pasar los datos por el modelo
        encoded = encoder(target)
        general_decoded = general_decoder(encoded)
        residual_decoded = residual_decoder(general_decoded)

        residual_decoded = residual_decoded.view(-1, 13, 13)
        target = target.view(-1, 13, 13)
        
        # Desescalar los datos
        outputs = scaler_output.inverse_transform(residual_decoded)
        target = scaler_output.inverse_transform(target)
        for i in range(5):
            visualizar_valores_pixeles(outputs[i], target[i])
            count += 1
        if count>= 5: break
        
# %%
