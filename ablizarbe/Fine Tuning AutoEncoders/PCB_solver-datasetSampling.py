#%%
import numpy as np
import runpy
from scipy import sparse
import matplotlib.pyplot as plt
import sys
import matplotlib.pyplot as plt
from matplotlib import colormaps
import torch
from torch.utils.data import Dataset
np.set_printoptions(threshold=sys.maxsize)


#####################################################################################################
######################################### PCB_case_1() ##############################################
#####################################################################################################

def PCB_case_1(L:float=0.1,thickness:float=0.001,m:int=3,board_k:float=10,ir_emmisivity:float=0.8,
                    T_interfaces:list=[250,250,250,250],Q_heaters:list=[1.0,1.0,1.0,1.0],Tenv:float=250,display:bool = False):
    """
    Caso 1. 
    PCB cuadrada de lado L con 4 heaters simétricamente colocados en coordenadas [(L/4,L/2),(L/2,L/4),(3*L/4,L/2),(L/2,3*L/4)]
    y con 4 nodos de interfaz situados en coordenadas [(0,0),(L,0),(L,L),(0,L)].
    Variables de entrada (unidades entre [], si no hay nada es adimensional):
                        -- L (int) = dimensiones de la placa. [m]
                        -- thickness (float) = espesor de la placa. [m]
                        -- m (int) = valor de refinamiento de malla. --> el número de nodos en x e y es n = 4*m+1. En el caso predeterminado son 12x12 nodos.
                        -- board_k (float) = conductividad térmica del material de la placa. [W/(K*m)]
                        -- ir_emmisivity (float) = emisividad infrarroja del recubrimiento óptico de la PCB (la pintura). 
                        -- T_interfaces (lista de 4 elementos) = temperatura de las 4 interfaces. [K]
                        -- Q_heaters (lista de 4 elementos) = potencia disipada por los heaters. [W]
                        -- Tenv (float) = temperatura del entorno. [K]
                        -- display (bool) = mostrar las temperaturas.
    Variables de salida:
                        -- T (numpy.array con dimension n = nx*ny) = vector con las temperaturas de los nodos (más información mirar en la descripción de **PCB_solver_main()**).
                        -- interfaces (diccionario {key = id del nodo, value = temperatura del nodo [K]}) = temperatura de las interfaces.
                        -- heaters (diccionario {key = id del nodo, value = disipación del nodo [W]}) = potencia disipada por los heaters.
    """

    n = 4*m+1

    id_Qnodes = [int((n-1)/4+(n-1)/2*n),int((n-1)/2+(n-1)/4*n),int(3*(n-1)/4+(n-1)/2*n),int((n-1)/2+3*(n-1)/4*n)]
    heaters = {id_Qnodes[0]:Q_heaters[0],id_Qnodes[1]:Q_heaters[1],id_Qnodes[2]:Q_heaters[2],id_Qnodes[3]:Q_heaters[3]}

    id_inodes = [0,n-1,n*n-1,n*n-n]
    interfaces = {id_inodes[0]:T_interfaces[0],id_inodes[1]:T_interfaces[1],id_inodes[2]:T_interfaces[2],id_inodes[3]:T_interfaces[3]}

    T = PCB_solver_main(Lx=L, Ly=L, thickness=thickness,nx=n,ny=n,board_k=board_k,ir_emmisivity=ir_emmisivity,
                    Tenv=Tenv,interfaces=interfaces,heaters=heaters, display=display)
    
    return T,interfaces,heaters


#####################################################################################################
######################################### PCB_case_2() ##############################################
#####################################################################################################

def PCB_case_2(L:float=0.1,thickness:float=0.001,m:int=3,board_k:float=10,ir_emmisivity:float=0.8,
                    T_interfaces:list=[250,250,250,250],Q_heaters:list=[1.0,1.0,1.0,1.0],Tenv:float=250,display:bool = False):
    """
    Caso 1
    PCB cuadrada de lado L con 4 heaters colocados en coordenadas [(L/4,L/2),(L/2,L/4),(L/4,3*L/4),(3*L/4,3*L/4)]
    y con 4 nodos de interfaz situados en coordenadas [(0,0),(L,0),(L,L),(0,L)].
    Variables de entrada (unidades entre [], si no hay nada es adimensional):
                        -- L (int) = dimensiones de la placa. [m]
                        -- thickness (float) = espesor de la placa. [m]
                        -- m (int) = valor de refinamiento de malla. --> el número de nodos en x e y es n = 4*m+1. En el caso predeterminado son 12x12 nodos.
                        -- board_k (float) = conductividad térmica del material de la placa. [W/(K*m)]
                        -- ir_emmisivity (float) = emisividad infrarroja del recubrimiento óptico de la PCB (la pintura). 
                        -- T_interfaces (lista de 4 elementos) = temperatura de las 4 interfaces (250 - 350 K). [K]
                        -- Q_heaters (lista de 4 elementos) = potencia disipada por los heaters (0.1 - 5.0 W). [W]
                        -- Tenv (float) = temperatura del entorno (250 - 350 K). [K]
                        -- display (bool) = mostrar las temperaturas.
    Variables de salida:
                        -- T (numpy.array con dimension n = nx*ny) = vector con las temperaturas de los nodos (más información mirar en la descripción de **PCB_solver_main()**).
                        -- interfaces (diccionario {key = id del nodo, value = temperatura del nodo [K]}) = temperatura de las interfaces.
                        -- heaters (diccionario {key = id del nodo, value = disipación del nodo [W]}) = potencia disipada por los heaters.
    """

    n = 4*m+1

    id_Qnodes = [int((n-1)/4+(n-1)/2*n),int((n-1)/2+(n-1)/4*n),int((n-1)/4+3*(n-1)/4*n),int(3*(n-1)/4+3*(n-1)/4*n)]
    heaters = {id_Qnodes[0]:Q_heaters[0],id_Qnodes[1]:Q_heaters[1],id_Qnodes[2]:Q_heaters[2],id_Qnodes[3]:Q_heaters[3]}

    id_inodes = [0,n-1,n*n-1,n*n-n]
    interfaces = {id_inodes[0]:T_interfaces[0],id_inodes[1]:T_interfaces[1],id_inodes[2]:T_interfaces[2],id_inodes[3]:T_interfaces[3]}

    T = PCB_solver_main(Lx=L, Ly=L, thickness=thickness,nx=n,ny=n,board_k=board_k,ir_emmisivity=ir_emmisivity,
                    Tenv=Tenv,interfaces=interfaces,heaters=heaters, display=display)
    
    return T,interfaces,heaters
    


#####################################################################################################
####################################### PCB_solver_main() ###########################################
#####################################################################################################

def PCB_solver_main(Lx:float,Ly:float,thickness:float,nx:int,ny:int,board_k:float,ir_emmisivity:float,
                    Tenv:float,interfaces:dict,heaters:dict, display:bool = False, maxiters:int = 1000, objtol:int = 0.01):
    '''
    Función solver del problema de PCB rectangular en un entorno radiativo formado por un cuerpo negro a temperatura Tenv. 
    Los nodos van numerados siguiendo el esquema de la figura, los nodos se ordenan de forma creciente filas.

    26---27---28---29---30    
    |    |    |    |    |  
    20---21---22---23---24    
    |    |    |    |    |   
    15---16---17---18---19    
    |    |    |    |    |
    10---11---12---13---14    
    |    |    |    |    |    y
    5----6----7----8----9    ^
    |    |    |    |    |    |
    0----1----2----3----4    ---> x

    Variables de entrada (unidades entre [], si no hay nada es adimensional):
                        -- Lx (int) = dimension x de la placa. [m]
                        -- Lx (int) = dimension y de la placa. [m]
                        -- thickness (float) = espesor de la placa. [m]
                        -- nx (int) = número de nodos en el eje x (en la figura de ejemplo son 5).
                        -- ny (int) = número de nodos en el eje y (en la figura de ejemplo son 6).
                        -- board_k (float) = conductividad térmica del material de la placa. [W/(K*m)]
                        -- ir_emmisivity (float) = emisividad infrarroja del recubrimiento óptico de la PCB (la pintura).
                        -- Tenv (float) = temperatura del entorno. [K]
                        -- interfaces (diccionario {key = id del nodo, value = temperatura del nodo [K]}) = temperatura de las interfaces.
                        -- heaters (diccionario {key = id del nodo, value = disipación del nodo [W]}) = potencia disipada por los heaters.
                        -- display (bool) = mostrar las temperaturas.
                        -- maxiters (int) = máximas iteraciones del solver. Mantener el valor predeterminado salvo si la convergencia es muy lenta (salta error en la linea 203). 
                        -- objtol (int) = tolerancia objetivo del solver. Mantener el valor predeterminado salvo si no se llega a convergencia (salta error en la linea 203).
    Variables de salida:
                        -- T (numpy.array con dimension n = nx*ny) = vector con las temperaturas ordenadas como en la figura de ejemplo.
    '''
    
    n_nodes = nx*ny # número total de nodos

    # cálculo de los GLs y GRs
    dx = Lx/(nx-1)
    dy = Ly/(ny-1)
    GLx = thickness*board_k*dy/dx
    GLy = thickness*board_k*dx/dy
    GR = 2*dx*dy*ir_emmisivity
    parameter = 2
    parameter2 = 0.5
    parameter3 = 1.2
    parameter4 = 0.8
    
    aroundheaters = []
    aroundheaters2 = []
    aroundheaters3 = []
    aroundheaters4 = []
    i = 1
    for id in heaters:
        if i == 1:
            aroundheaters.append(id)
            aroundheaters.append(id+1)
            aroundheaters.append(id-1)
            aroundheaters.append(id+nx)
            aroundheaters.append(id-nx)
        elif i == 2:
            aroundheaters2.append(id)
            aroundheaters2.append(id+1)
            aroundheaters2.append(id-1)
            aroundheaters2.append(id+nx)
            aroundheaters2.append(id-nx)
        elif i == 3:
            aroundheaters3.append(id)
            aroundheaters3.append(id+1)
            aroundheaters3.append(id-1)
            aroundheaters3.append(id+nx)
            aroundheaters3.append(id-nx)
        elif i == 4:
            aroundheaters4.append(id)
            aroundheaters4.append(id+1)
            aroundheaters4.append(id-1)
            aroundheaters4.append(id+nx)
            aroundheaters4.append(id-nx)
        i += 1

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
            elif id in aroundheaters:
                GLii = 0
                if i+1 < nx:
                    K_rows.append(id)
                    K_cols.append(id+1)
                    K_data.append(-GLx* parameter)
                    GLii += GLx * parameter
                if i-1 >= 0:
                    K_rows.append(id)
                    K_cols.append(id-1)
                    K_data.append(-GLx* parameter)
                    GLii += GLx * parameter
                if j+1 < ny:
                    K_rows.append(id)
                    K_cols.append(id+nx)
                    K_data.append(-GLx* parameter)
                    GLii += GLy * parameter
                if j-1 >= 0:
                    K_rows.append(id)
                    K_cols.append(id-nx)
                    K_data.append(-GLx* parameter)
                    GLii += GLy * parameter
                K_rows.append(id)
                K_cols.append(id)
                K_data.append(GLii)
            elif id in aroundheaters2:
                GLii = 0
                if i+1 < nx:
                    K_rows.append(id)
                    K_cols.append(id+1)
                    K_data.append(-GLx* parameter2)
                    GLii += GLx * parameter2
                if i-1 >= 0:
                    K_rows.append(id)
                    K_cols.append(id-1)
                    K_data.append(-GLx* parameter2)
                    GLii += GLx * parameter2
                if j+1 < ny:
                    K_rows.append(id)
                    K_cols.append(id+nx)
                    K_data.append(-GLx* parameter2)
                    GLii += GLy * parameter2
                if j-1 >= 0:
                    K_rows.append(id)
                    K_cols.append(id-nx)
                    K_data.append(-GLx* parameter2)
                    GLii += GLy * parameter2
                K_rows.append(id)
                K_cols.append(id)
                K_data.append(GLii)
            elif id in aroundheaters3:
                GLii = 0
                if i+1 < nx:
                    K_rows.append(id)
                    K_cols.append(id+1)
                    K_data.append(-GLx* parameter3)
                    GLii += GLx * parameter3
                if i-1 >= 0:
                    K_rows.append(id)
                    K_cols.append(id-1)
                    K_data.append(-GLx* parameter3)
                    GLii += GLx * parameter3
                if j+1 < ny:
                    K_rows.append(id)
                    K_cols.append(id+nx)
                    K_data.append(-GLx* parameter3)
                    GLii += GLy * parameter3
                if j-1 >= 0:
                    K_rows.append(id)
                    K_cols.append(id-nx)
                    K_data.append(-GLx* parameter3)
                    GLii += GLy * parameter3
                K_rows.append(id)
                K_cols.append(id)
                K_data.append(GLii)
            elif id in aroundheaters4:
                GLii = 0
                if i+1 < nx:
                    K_rows.append(id)
                    K_cols.append(id+1)
                    K_data.append(-GLx* parameter4)
                    GLii += GLx * parameter4
                if i-1 >= 0:
                    K_rows.append(id)
                    K_cols.append(id-1)
                    K_data.append(-GLx* parameter4)
                    GLii += GLx * parameter4
                if j+1 < ny:
                    K_rows.append(id)
                    K_cols.append(id+nx)
                    K_data.append(-GLx* parameter4)
                    GLii += GLy * parameter4
                if j-1 >= 0:
                    K_rows.append(id)
                    K_cols.append(id-nx)
                    K_data.append(-GLx* parameter4)
                    GLii += GLy * parameter4
                K_rows.append(id)
                K_cols.append(id)
                K_data.append(GLii)
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
    K = sparse.csr_matrix((K_data,(K_rows,K_cols)),shape=(n_nodes,n_nodes))


    # Creación de la matriz de acoplamientos radiativos [E]
    E_data = []
    E_id = []
    for id in range(n_nodes):
        if id not in interfaces:
            E_id.append(id)
            E_data.append(GR)
    E = sparse.csr_matrix((E_data,(E_id,E_id)),shape=(n_nodes,n_nodes))

    # Creación del vector {Q}.
    Q = np.zeros(n_nodes,dtype=np.double)
    for id in range(n_nodes):
        if id in interfaces:
            Q[id] = interfaces[id]
        elif id in heaters:
            Q[id] = heaters[id]
    
    # Resolución de la ecuación no lineal [K]{T} + Boltzmann_cte*[E]({T^4} - Tenv^4) = {Q} 
    # mediante la resolución iterativa de la ecuación [A]{dT_i} = {b}, donde:
    #           -- [A] = [K] + 4*Boltzmann_cte*[E].*{T_i^3} (.* = multiplicación elemento a elemento)
    #           -- {b} = {Q} - [K]*{T_i} - [E]*({T_i^4}-Tenv^4)
    #           -- {T_i+1} = {T_i} + {dT_i}
            
    Boltzmann_cte = 5.67E-8
    tol = 100
    it = 0
    T = np.full(n_nodes,Tenv,dtype=np.double)

    while tol > objtol and it < maxiters:
        b = Q-K.__matmul__(T) - Boltzmann_cte*E.__matmul__(T**4-Tenv**4)
        A = K+4*Boltzmann_cte*E.multiply(T**3)
        dT = sparse.linalg.spsolve(A,b)
        T += dT
        tol = max(abs(dT))
        it = it+1

    if tol > 0.01:
        print("ERROR in PCB SOLVER MAIN. Convergence was not reached.")
        exit(1)
    
    if display == True:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6), constrained_layout=True)
        psm = ax.pcolormesh(T.reshape(ny,nx), cmap=colormaps['jet'], rasterized=True, vmin=np.min(T), vmax=np.max(T))
        fig.colorbar(psm, ax=ax)
        plt.title('Temperature field')
        plt.show()
    
    return T



##########################################CREACIÓN DE LA CLASE########################################

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

    


######################################################
########### DISTRIBUCIÓN UNIFORME DE DATOS ###########
######################################################

import numpy as np
from scipy.optimize import differential_evolution
from collections import Counter

# Define the ranges for each variable
temp_min, temp_max = 250, 350
power_min, power_max = 0.1, 5.0

total_UNeterror = []
total_CNNerror = []
total_Linearerror = []

for n_samples in[5,10,13,15,18,20,25,30,40]:

    print("Number of samples: ", n_samples)

    UNeterror = []
    Linearerror = []
    CNNerror = []

    for _ in range(5):
        # Number of variables and samples
        n_variables = 9
        #n_samples = 20  # Choose the number of samples you need

        results_list = []

        for i in range(25):

            # Objective function to maximize minimum distance between points
            def min_distance_objective(samples):
                samples = samples.reshape((n_samples, n_variables))
                dist = np.min(np.linalg.norm(samples[:, np.newaxis, :] - samples[np.newaxis, :, :], axis=2))
                return -dist

            # Bounds for the optimizer
            bounds = [(0, 1)] * (n_samples * n_variables)

            # Perform differential evolution
            result = differential_evolution(min_distance_objective, bounds, maxiter=1000)
            optimized_samples = result.x.reshape((n_samples, n_variables))

            results_list.append(tuple(map(tuple, optimized_samples)))

        # Count the occurrences of each unique result
        counter = Counter(results_list)

        # Find the most common result
        most_common_result, count = counter.most_common(1)[0]

        # Convert the result back to a NumPy array
        optimized_samples = np.array(most_common_result)


        # Scale the samples to the appropriate ranges
        scaled_samples = np.zeros_like(optimized_samples)

        for i in range(4):  # Power dissipators
            scaled_samples[:, i] = optimized_samples[:, i] * (power_max - power_min) + power_min

        for i in range(4, 8):  # Interface temperatures
            scaled_samples[:, i] = optimized_samples[:, i] * (temp_max - temp_min) + temp_min

        scaled_samples[:, 8] = optimized_samples[:, 8] * (temp_max - temp_min) + temp_min

        scaled_samples_output = []

        for i in range(n_samples):
            # Obtain the results from the function
            resultados, __, _ = PCB_case_2(T_interfaces=scaled_samples[i,4:8], Q_heaters=scaled_samples[i,0:4], Tenv=scaled_samples[i,8])
            resultados = resultados.reshape(13, 13)

            # Add the result matrices to the data
            scaled_samples_output.append(resultados)

        scaled_samples_output = np.array(scaled_samples_output)

        extra = np.zeros(n_samples)    


        ##############################################################
        ################# CREACIÓN DEL DATASET #######################
        ##############################################################

        n_entradas = 1000
        nodos_lado = 13

        input = []
        output = []

        #Generación de datos aleatorios para el dataset
        potenciasAleatorias = np.random.uniform(0.1, 5, (n_entradas, 4))
        interfacesAleatorias = np.random.uniform(250, 350, (n_entradas, 4))
        TenvAleatorias = np.random.uniform(250, 350, n_entradas)


        for i in range(n_entradas):

            # Obtener los resultados de la función
            resultados,__,_ = PCB_case_2(T_interfaces=interfacesAleatorias[i],Q_heaters=potenciasAleatorias[i],Tenv=TenvAleatorias[i])
            resultados = resultados.reshape(nodos_lado,nodos_lado)

            #Añadimos a los datos las matrices de resultados
            output.append(resultados)

            input1 = []
            input1 = np.concatenate((potenciasAleatorias[i],interfacesAleatorias[i],TenvAleatorias[i]), axis=None)
            
            
            #Añadimos a los datos las matrices de entrada

            input.append(input1)



        #Convertimos a arrays
        input = np.array(input)
        input = np.concatenate((scaled_samples, input), axis=0)
        input = torch.tensor(input, dtype=torch.float32)


        scalar_input = np.array(TenvAleatorias)
        scalar_input = np.concatenate((extra, scalar_input), axis=0)
        scalar_input = torch.tensor(scalar_input, dtype=torch.float32)

        output = np.array(output)
        output = np.concatenate((scaled_samples_output, output), axis=0)
        output = torch.tensor(output, dtype=torch.float32)

        #Guardamos el dataset
        dataset = PCBDataset(input, output, scalar_input)
        torch.save(dataset, 'PCB_dataset_modified.pth')




        ##############################################################
        ################# CREACIÓN DEL DATASET #######################
        ##############################################################

        n_entradas = 1000
        nodos_lado = 13

        input = []
        output = []

        #Generación de datos aleatorios para el dataset
        potenciasAleatorias = np.random.uniform(0.1, 5, (n_entradas, 4))
        interfacesAleatorias = np.random.uniform(250, 350, (n_entradas, 4))
        TenvAleatorias = np.random.uniform(250, 350, n_entradas)

        for i in range(n_samples):
            #Convertimos en matrices las potencias y temperaturas de las interfaces
            potencias = np.zeros((nodos_lado,nodos_lado))
            interfaces = np.zeros((nodos_lado,nodos_lado))

            potencias[6,3], potencias[3,6], potencias[9,3], potencias[9,9] =  scaled_samples[i,0:4]
            interfaces[0,0], interfaces[0,nodos_lado-1], interfaces[nodos_lado-1,nodos_lado-1], interfaces[nodos_lado-1,0] =  scaled_samples[i,4:8]

            #Añadimos a los datos las matrices de entrada
            input1 = []
            input1.append(potencias)
            input1.append(interfaces)
            input.append(input1)


        for i in range(n_entradas):

            # Obtener los resultados de la función
            resultados,__,_ = PCB_case_2(T_interfaces=interfacesAleatorias[i],Q_heaters=potenciasAleatorias[i],Tenv=TenvAleatorias[i])
            resultados = resultados.reshape(nodos_lado,nodos_lado)

            #Añadimos a los datos las matrices de resultados
            output.append(resultados)

            #Convertimos en matrices las potencias y temperaturas de las interfaces
            potencias = np.zeros((nodos_lado,nodos_lado))
            interfaces = np.zeros((nodos_lado,nodos_lado))

            potencias[6,3], potencias[3,6], potencias[9,3], potencias[9,9] = potenciasAleatorias[i]
            interfaces[0,0], interfaces[0,nodos_lado-1], interfaces[nodos_lado-1,nodos_lado-1], interfaces[nodos_lado-1,0] = interfacesAleatorias[i]

            #Añadimos a los datos las matrices de entrada
            input1 = []
            input1.append(potencias)
            input1.append(interfaces)
            input.append(input1)



        #Convertimos a arrays
        input = np.array(input)
        input = torch.tensor(input, dtype=torch.float32)

        scalar_input = np.array(TenvAleatorias)
        scalar_input = np.concatenate((scaled_samples[:,8], scalar_input), axis=0)
        scalar_input = torch.tensor(scalar_input, dtype=torch.float32)

        output = np.array(output)
        output = np.concatenate((scaled_samples_output, output), axis=0)
        output = torch.tensor(output, dtype=torch.float32)

        #Guardamos el dataset
        dataset = PCBDataset(input, output, scalar_input)
        torch.save(dataset, 'PCB_dataset_modified2.pth')


        variables = {"train_cases": n_samples}

        globals_dict = runpy.run_path("finetune_UNet.py", init_globals=variables)

        UNeterror.append(globals_dict["avg_test_loss"])

        UNetTime = globals_dict["total_time"]

        globals_dict = runpy.run_path("autoDavidBucle.py", init_globals=variables)

        CNNerror.append(globals_dict["avg_test_loss"])

        CNNTime = globals_dict["total_time"]

        globals_dict = runpy.run_path("autoencoderBucle.py", init_globals=variables)

        Linearerror.append(globals_dict["avg_test_loss"])

        LinearTime = globals_dict["total_time"]
        


    total_UNeterror.append(UNeterror)
    total_CNNerror.append(CNNerror)
    total_Linearerror.append(Linearerror)
    print("----------------------------------")
    print("UNet error: ", np.mean(UNeterror), "Time: ", int(UNetTime))
    print("Std UNet error: ", np.std(UNeterror))
    print("Min UNet error: ", np.min(UNeterror))
    print("----------------------------------")
    print("CNN error: ", np.mean(CNNerror), "Time: ", int(CNNTime))
    print("Std CNN error: ", np.std(CNNerror))
    print("Min CNN error: ", np.min(CNNerror))
    print("----------------------------------")
    print("Linear error: ", np.mean(Linearerror), "Time: ", int(LinearTime))
    print("Min Linear error: ", np.min(Linearerror))
    print("Std Linear error: ", np.std(Linearerror))
    print("----------------------------------")
    print("----------------------------------")


print("Total UNet error: ", total_UNeterror)
print("Total CNN error: ", total_CNNerror)
print("Total Linear error: ", total_Linearerror)
# %%
