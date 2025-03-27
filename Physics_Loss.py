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
                        K_data.append(-GLy)
                        GLii += GLy
                    if j-1 >= 0:
                        K_rows.append(id)
                        K_cols.append(id-nx)
                        K_data.append(-GLy)
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
    def __init__(self, 
                 L: float = 0.1,
                 thickness: float = 0.001,
                 board_k: float = 15,
                 ir_emmisivity: float = 0.8,
                 board_rho: float = 2700,
                 board_c: float = 900,
                 dt: float = 1
                 ):
        super(PhysicsLossTransient, self).__init__()

        nx = 13
        ny = 13
        
        self.n_nodes = nx*ny # número total de nodos
        
        interfaces = [0,nx-1,nx*nx-1,nx*nx-nx]

        self.Boltzmann_cte = 5.67E-8
        self.rho = board_rho
        self.cp = board_c
        self.dt = dt

        # cálculo de los GLs y GRs
        self.dx = L/(nx-1)
        self.dy = L/(ny-1)
        GLx = thickness*board_k*self.dy/self.dx
        GLy = thickness*board_k*self.dx/self.dy
        GR = 2*self.dx*self.dy*ir_emmisivity
        
        self.thickness = thickness

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
                        K_data.append(-GLy)
                        GLii += GLy
                    if j-1 >= 0:
                        K_rows.append(id)
                        K_cols.append(id-nx)
                        K_data.append(-GLy)
                        GLii += GLy
                    K_rows.append(id)
                    K_cols.append(id)
                    K_data.append(GLii)
                    
        indices = torch.LongTensor([K_rows, K_cols])
        values = torch.FloatTensor(K_data)
        shape = torch.Size([self.n_nodes, self.n_nodes])
        self.K = torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float).cuda()

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


    def forward(self, 
                T_new,     # Temperatura en el instante n+1 (predicha por la red, por ej)
                T_old,     # Temperatura en el instante n (condición conocida), 
                heaters_input, 
                interfaces_input, 
                Tenv):
        """
        T_new: tensor [batch_size, n_nodes] con la T en el paso n+1
        T_old: tensor [batch_size, n_nodes] con la T del paso n
        heaters, interfaces, Tenv: análogos a lo anterior.
        
        Devuelve la norma del 'residuo transitorio' según:
            rho*c*(espesor)*(dx*dy)*(T_new - T_old)/dt  -  [ Q - K*T_old - sigma*E*(T_old^4 - Tenv^4) ]
        """

        #Generación del vector T
        T_new = T_new.flatten(start_dim=1).transpose(0,1).cuda()  # shape => [n_nodes, batch_size]
        T_old = T_old.flatten(start_dim=1).transpose(0,1).cuda()
        Tenv  = Tenv.flatten(start_dim=1).transpose(0,1).cuda()
        
        Q = (heaters_input + interfaces_input).float().flatten(start_dim=1).transpose(0,1).cuda()
        
        # Calculamos lado derecho del balance (igual que en el solver)
        # [Q - K*T_old - sigma*E*(T_old^4 - Tenv^4)]
        rhs =   Q \
                - torch.sparse.mm(self.K, T_old) \
                - self.Boltzmann_cte * torch.sparse.mm(self.E, (torch.pow(T_old, 4) - torch.pow(Tenv, 4)))
                  
        # Calculamos lado izquierdo transitorio:
        #  rho*c*(espesor)*(dx*dy)*(T_new - T_old)/dt
        vol_heat = self.rho * self.cp * self.thickness * (self.dx * self.dy)
        lhs_trans = vol_heat * (T_new - T_old) / self.dt

        # ExcessEnergy = LHS - RHS
        residual = lhs_trans - rhs  # [n_nodes, batch_size]

        return torch.mean(torch.abs(residual))