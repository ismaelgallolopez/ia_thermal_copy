import torch
import torch.nn as nn

class LaEnergiaNoAparece(nn.Module):
    def __init__(self, L:float=0.1,thickness:float=0.001,board_k:float=15,ir_emmisivity:float=0.8):
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


    def forward(self, outputs, heaters, interfaces, Tenv):
        
        #Generación del vector Q
        heaters = torch.flatten(heaters, start_dim=1)

        interfaces = torch.flatten(interfaces, start_dim=1)

        Q = heaters + interfaces
        

        #Generación del vector T
        T = torch.flatten(outputs, start_dim=1)

        excessEnergy = torch.zeros((self.n_nodes,Q.size(0)))


        T, Q, Tenv = T.cuda(), Q.cuda(), Tenv.cuda()
        T = torch.transpose(T, 0, 1)
        Q = torch.transpose(Q, 0, 1)
        Tenv = torch.transpose(Tenv, 0, 1)

        excessEnergy = torch.sparse.mm(self.K,T) + self.Boltzmann_cte*torch.sparse.mm(self.E,(T**4-Tenv**4))-Q

        return torch.mean(torch.abs(excessEnergy))
    
    
    
class PhysicsLossTransient(nn.Module):
    def __init__(self, L=0.1, thickness=0.001, board_k=15, ir_emmisivity=0.8,
                 rho=2700, cp=900, nx=13, ny=13):
        super(PhysicsLossTransient, self).__init__()

        self.nx = nx
        self.ny = ny
        self.n_nodes = nx * ny

        self.L = L
        self.thickness = thickness
        self.rho = rho
        self.cp = cp
        self.dx = L / (nx - 1)
        self.dy = L / (ny - 1)
        self.Boltzmann_cte = 5.67e-8

        # Interfaces de contorno (esquinas)
        interfaces = [0, nx-1, nx*ny-1, nx*ny-nx]

        # Coeficientes de conductividad y radiación
        GLx = thickness * board_k * self.dy / self.dx
        GLy = thickness * board_k * self.dx / self.dy
        GR = 2 * self.dx * self.dy * ir_emmisivity

        # Matriz de conductividad [K]
        K_rows = []
        K_cols = []
        K_data = []
        for j in range(ny):
            for i in range(nx):
                id = i + nx * j
                if id in interfaces:
                    K_rows.append(id)
                    K_cols.append(id)
                    K_data.append(1.0)
                else:
                    GLii = 0.0
                    if i + 1 < nx:
                        K_rows.append(id)
                        K_cols.append(id + 1)
                        K_data.append(-GLx)
                        GLii += GLx
                    if i - 1 >= 0:
                        K_rows.append(id)
                        K_cols.append(id - 1)
                        K_data.append(-GLx)
                        GLii += GLx
                    if j + 1 < ny:
                        K_rows.append(id)
                        K_cols.append(id + nx)
                        K_data.append(-GLy)
                        GLii += GLy
                    if j - 1 >= 0:
                        K_rows.append(id)
                        K_cols.append(id - nx)
                        K_data.append(-GLy)
                        GLii += GLy
                    K_rows.append(id)
                    K_cols.append(id)
                    K_data.append(GLii)

        indices = torch.LongTensor([K_rows, K_cols])
        values = torch.FloatTensor(K_data)
        shape = torch.Size([self.n_nodes, self.n_nodes])
        self.K = torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float).cuda()

        # Matriz de radiación [E]
        E_data = []
        E_id = []
        for id in range(self.n_nodes):
            if id not in interfaces:
                E_id.append(id)
                E_data.append(GR)
        indices = torch.tensor([E_id, E_id], dtype=torch.int64)
        values = torch.tensor(E_data, dtype=torch.float32)
        size = torch.Size([self.n_nodes, self.n_nodes])
        self.E = torch.sparse_coo_tensor(indices, values, size).cuda()

    def forward(self, T_pred, heaters, interfaces, Tenv, T_prev, dt):
        """
        Calcula el residuo del balance térmico transitorio.

        T_pred: [batch, nx, ny]  - temperatura predicha en t + dt
        T_prev: [batch, nx, ny]  - temperatura en t
        heaters: [batch, nx, ny] - fuente térmica
        interfaces: [batch, nx, ny] - potencia impuesta en interfaces
        Tenv: [batch, nx, ny] - entorno radiativo
        dt: float - intervalo de tiempo
        """

        # Aplanar y transponer para que queden como [n_nodes, batch]
        T_pred = torch.flatten(T_pred, start_dim=1).T.cuda()
        T_prev = torch.flatten(T_prev, start_dim=1).T.cuda()
        Q = torch.flatten(heaters + interfaces, start_dim=1).T.cuda()
        Tenv = torch.flatten(Tenv, start_dim=1).T.cuda()

        # Derivada temporal modelada por la red
        dTdt_model = (T_pred - T_prev) / dt

        # Término físico de la ecuación de calor
        conduction_term = torch.sparse.mm(self.K, T_prev)
        radiation_term = self.Boltzmann_cte * torch.sparse.mm(self.E, T_prev**4 - Tenv**4)
        rhs_physics = (Q - conduction_term - radiation_term) / (self.rho * self.cp * self.thickness * self.dx * self.dy)

        # Residuo
        residual = dTdt_model - rhs_physics

        # Enmascarar nodos de interfaz (anular derivada)
        interface_mask = (interfaces != 0).float().T.cuda()
        residual = residual * (1 - interface_mask)

        return torch.mean(torch.abs(residual))