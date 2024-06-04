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
scaler_input = StandardScaler3D()
scaler_scalar = GlobalStandardScaler()
scaler_output = GlobalStandardScaler()

scaled_input = scaler_input.fit_transform(dataset.inputs_dataset)
scaled_scalar = scaler_scalar.fit_transform(dataset.scalar_dataset)
scaled_output = scaler_output.fit_transform(dataset.outputs_dataset)


dataset = PCBDataset(scaled_input, scaled_output, scaled_scalar)

# Separando Train and Test
train_cases = 1000
test_cases = 1000

batch_size = 100
test_size = 0.1
#num_train = test_cases + train_cases 
num_train = int(len(dataset))
indices = list(range(num_train))

np.random.seed(42)
np.random.shuffle(indices)
#split = int(np.floor(test_cases))
split = int(np.floor(test_size * num_train))
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
        self.fc1 = nn.Linear(13*13, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 12)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        return x

# Definir el Decodificador General (GCD)
class GeneralDecoder(nn.Module):
    def __init__(self):
        super(GeneralDecoder, self).__init__()
        self.fc1 = nn.Linear(12, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 13*13)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Definir el Decodificador Residual (RCD)
class ResidualDecoder(nn.Module):
    def __init__(self):
        super(ResidualDecoder, self).__init__()
        self.fc1 = nn.Linear(13*13, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 13*13)

    def forward(self, x):
        input = x
        residual = F.leaky_relu(self.fc1(x))
        residual = F.leaky_relu(self.fc2(residual))
        residual = F.leaky_relu(self.fc3(residual))
        residual = self.fc4(residual)
        return input + residual  # Sumar el input con el residuo para obtener el output

# Definir el Discriminador para ACD
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.adaptative_pool = nn.AdaptiveMaxPool2d((8, 8))
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*2*2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = self.adaptative_pool(x)
        x = F.leaky_relu(self.conv2(x))
        x = self.pool(x)
        x = F.leaky_relu(self.conv3(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

# Definir los modelos
encoder = Encoder()
general_decoder = GeneralDecoder()
residual_decoder = ResidualDecoder()
discriminator = Discriminator()

# Definir los optimizadores
optim_generator = optim.Adam(list(encoder.parameters()) + list(general_decoder.parameters()) + list(residual_decoder.parameters()), lr=0.0001)
optim_discriminator = optim.Adam(discriminator.parameters(), lr=0.0001)

# Definir las funciones de pérdida
criterionReconstruction = nn.MSELoss()
criterionDiscriminator = nn.BCELoss()

last_test_loss = np.inf

#%%

######################################
############# TRAIN LOOP #############
######################################


# Número de épocas
num_epochs = 200

# Ruta para guardar los modelos
base_path = 'modelos\modelo9_{}.pth'

models = {
    "encoder": encoder,
    "generalDecoder": general_decoder,
    "residualDecoder": residual_decoder,
    "discriminator": discriminator
}

# Precálculo de etiquetas reales y falsas fuera del bucle de entrenamiento
real_label = 1
fake_label = 0

# Suponiendo que batch_size es constante, de lo contrario, ajustar dentro del bucle
fixed_batch_size = next(iter(train_loader))[2].size(0)
real_labels = torch.full((fixed_batch_size, 1), real_label, dtype=torch.float)
fake_labels = torch.full((fixed_batch_size, 1), fake_label, dtype=torch.float)

# Enviar etiquetas al dispositivo correcto, por ejemplo 'cuda' si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
real_labels, fake_labels = real_labels.to(device), fake_labels.to(device)
encoder, general_decoder, residual_decoder, discriminator = encoder.to(device), general_decoder.to(device), residual_decoder.to(device), discriminator.to(device)

for epoch in range(num_epochs):

    encoder.train()
    general_decoder.train()
    residual_decoder.train()

    for _, target, _ in train_loader: 

        target = target.view(-1, 13*13).to(device)
        batch_size = target.size(0)
        
        # Asegurar que las etiquetas tienen el tamaño correcto si batch_size varía
        if batch_size != fixed_batch_size:
            real_labels = torch.full((batch_size, 1), real_label, dtype=torch.float, device=device)
            fake_labels = torch.full((batch_size, 1), fake_label, dtype=torch.float, device=device)
            fixed_batch_size = batch_size
        
        ### Entrenamiento del generador y encoder
        optim_generator.zero_grad()
        
        encoded = encoder(target)
        general_decoded = general_decoder(encoded)
        residual_decoded = residual_decoder(general_decoded)
        
        gen_loss = criterionReconstruction(residual_decoded, target)
        validity = discriminator(residual_decoded.view(-1, 1, 13, 13))
        adversarial_loss = criterionDiscriminator(validity, real_labels)
        
        g_loss = gen_loss + adversarial_loss
        g_loss.backward()
        optim_generator.step()
        
        ### Entrenamiento del discriminador
        optim_discriminator.zero_grad()
        
        with torch.no_grad(): 
            fake_data = residual_decoded.detach()
        
        real_pred = discriminator(target.view(-1, 1, 13, 13))
        d_real_loss = criterionDiscriminator(real_pred, real_labels)
        
        fake_pred = discriminator(fake_data.view(-1, 1, 13, 13))
        d_fake_loss = criterionDiscriminator(fake_pred, fake_labels)
        
        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward()
        optim_discriminator.step()
    
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
        
        print(f"Epoch {epoch+1}/{num_epochs}, Generator Loss: {g_loss.item():.8f}, Discriminator Loss: {d_loss.item():.8f}")
        
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
base_path = 'modelos\modelo9_{}.pth'

models = {
    "encoder": encoder,
    "generalDecoder": general_decoder,
    "residualDecoder": residual_decoder,
    "discriminator": discriminator
}

for name, model in models.items():
    model_path = base_path.format(name)
    model.load_state_dict(torch.load(model_path))


# Enviar etiquetas al dispositivo correcto, por ejemplo 'cuda' si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder, general_decoder, residual_decoder, discriminator = encoder.to(device), general_decoder.to(device), residual_decoder.to(device), discriminator.to(device)

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
######################################
############# TEST LOOP ##############
######################################

# Enviar etiquetas al dispositivo correcto, por ejemplo 'cuda' si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder, general_decoder, residual_decoder, discriminator = encoder.to(device), general_decoder.to(device), residual_decoder.to(device), discriminator.to(device)

    
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
encoder, general_decoder, residual_decoder, discriminator = encoder.to(device), general_decoder.to(device), residual_decoder.to(device), discriminator.to(device)


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
            text = axs[1].text(j, i, f'{output_np[i, j]:.0f}',
                               ha="center", va="center", color="w", fontsize=6)

    # Target
    axs[0].imshow(target_np, cmap='viridis', interpolation='nearest')
    axs[0].title.set_text('Target')
    for i in range(target_np.shape[0]):
        for j in range(target_np.shape[1]):
            text = axs[0].text(j, i, f'{target_np[i, j]:.0f}',
                               ha="center", va="center", color="w", fontsize=6)

    plt.show()

count = 0 
with torch.no_grad(): 
    for batch, target, scalar in test_loader:

        # Prepare the data and target
        target = target.view(-1, 13*13).to(device)

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
