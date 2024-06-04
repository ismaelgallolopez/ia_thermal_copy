#%%

import chess
import chess.pgn
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv1 = nn.Conv2d(15, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 1024)
        # Supongamos que reducimos todos los movimientos posibles a un tamaño fijo.
        # que es más que el número total de movimientos posibles en cualquier estado dado del tablero de ajedrez.
        self.fc2 = nn.Linear(1024, 4160) 

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(-1, 128 * 8 * 8)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    


def tablero_a_tensor(tablero):
    # 6 tipos de piezas * 2 colores + 2 (enroque) + 1 (al paso) = 15 canales
    tensor = np.zeros((15, 8, 8), dtype=np.float32)
    
    pieza_a_canal = {
        chess.PAWN: 0,
        chess.KNIGHT: 2,
        chess.BISHOP: 4,
        chess.ROOK: 6,
        chess.QUEEN: 8,
        chess.KING: 10
    }
    
    for i in range(64):
        pieza = tablero.piece_at(i)
        if pieza:
            canal = pieza_a_canal[pieza.piece_type] + (pieza.color == chess.BLACK)
            fila, columna = divmod(i, 8)
            tensor[canal, fila, columna] = 1
    
    # Enroque
    tensor[12, :, :] = tablero.has_kingside_castling_rights(chess.WHITE)
    tensor[13, :, :] = tablero.has_kingside_castling_rights(chess.BLACK)
    # Al paso
    if tablero.ep_square is not None:
        fila, columna = divmod(tablero.ep_square, 8)
        tensor[14, fila, columna] = 1
    
    return tensor



def movimiento_a_indice(movimiento):
    origen = movimiento.from_square
    destino = movimiento.to_square
    promocion = movimiento.promotion
    
    indice_base = origen * 64 + destino  # 4096 posibles movimientos sin promoción
    
    if promocion:
        promocion_a_indice = {chess.QUEEN: 0, chess.ROOK: 1, chess.BISHOP: 2, chess.KNIGHT: 3}
        indice_promocion = promocion_a_indice[promocion]
        
        # Calcular índices para promociones en la octava fila para las blancas o la primera fila para las negras
        if destino >= 56 and destino <= 63:  # Promociones en la octava fila
            indice_base = 64 * 64 + (destino - 56) * 4 + indice_promocion
        elif destino >= 0 and destino <= 7:  # Promociones en la primera fila
            indice_base = 64 * 64 + (destino) * 4 + indice_promocion
    
    return indice_base



def seleccionar_mejor_movimiento(modelo, tablero):
    # Convertir el tablero a la representación de entrada adecuada para el modelo
    entrada = tablero_a_tensor(tablero)  # Necesitas implementar esta función
    entrada = torch.tensor(entrada, dtype=torch.float32).unsqueeze(0)
    entrada = entrada.cuda()
    with torch.no_grad():
        predicciones = modelo(entrada)
    probabilidades = torch.softmax(predicciones, dim=1)
    
    # Obtener los movimientos legales
    movimientos_legales = list(tablero.legal_moves)
    mejor_movimiento = None
    max_probabilidad = -float("inf")

    # Buscar el movimiento legal con la mayor probabilidad
    for movimiento in movimientos_legales:
        indice_movimiento = movimiento_a_indice(movimiento)  # Implementa esta función
        probabilidad = probabilidades[0,indice_movimiento]
        if probabilidad > max_probabilidad:
            mejor_movimiento = movimiento
            max_probabilidad = probabilidad

    return mejor_movimiento

def calcular_recompensa_final(tablero):
    resultado = tablero.result()
    if resultado == "1-0":  # Ganan las blancas
        return 1 if tablero.turn == chess.WHITE else -0.2
    elif resultado == "0-1":  # Ganan las negras
        return -0.2 if tablero.turn == chess.WHITE else 1
    else:  # Empate
        return -1
    
def calcular_perdida(model, estado, accion, recompensa_final):
    # Asumimos que el modelo predice un valor que representa la probabilidad estimada de ganar
    estado = estado.cuda()
    predicción = model(estado).squeeze(0)
    
    # Convertimos la acción a una representación one-hot para seleccionar la predicción correspondiente
    accion_one_hot = torch.zeros(4160)  # Asume 4160 movimientos posibles
    accion_one_hot[accion] = 1
    accion_one_hot = accion_one_hot.cuda()
    predicción_acción = torch.dot(predicción, accion_one_hot)
    
    # La recompensa final es +1 (ganar), 0 (empatar), o -1 (perder)
    # Convertimos esta recompensa en una probabilidad objetivo
    # +1 (ganar) -> 1.0, 0 (empatar) -> 0.5, -1 (perder) -> 0.0
    probabilidad_objetivo = (recompensa_final + 1) / 2.0
    
    # Calculamos la pérdida como la diferencia cuadrada entre la predicción y el objetivo
    perdida = (predicción_acción - probabilidad_objetivo) ** 2
    
    return perdida

def jugar_partida(modelo):
    tablero = chess.Board()
    juego = chess.pgn.Game()
    juego.headers["Event"] = "Self play"
    juego.headers["Site"] = "Local"
    juego.headers["Date"] = "2024.02.25"
    juego.headers["Round"] = "1"
    juego.headers["White"] = "ChessNet"
    juego.headers["Black"] = "ChessNet"
    node = juego

    # Mientras no haya terminado la partida
    while not tablero.is_game_over():
        # Seleccionar el mejor movimiento con el modelo actual
        movimiento = seleccionar_mejor_movimiento(modelo, tablero)
        node = node.add_variation(movimiento)
        tablero.push(movimiento)

    # Establecer el resultado de la partida
    juego.headers["Result"] = tablero.result()

    # Convertir el objeto Game a PGN
    exportador = chess.pgn.StringExporter(headers=True, variations=True, comments=False)
    pgn_string = juego.accept(exportador)
    return pgn_string

model = ChessNet()
model = model.cuda()
optimizer = optim.Adam(model.parameters(), lr = 0.001)


#%%
#################################
############# TRAIN #############
#################################
num_episodios = 1000


for episodio in range(num_episodios):
    tablero = chess.Board()
    historial_estados = []
    historial_acciones = []

    while not tablero.is_game_over():
        # Convertir el estado actual del tablero a tensor
        estado_actual = tablero_a_tensor(tablero)  # Añadir dimensión de lote
        estado_actual = torch.tensor(estado_actual, dtype=torch.float32).unsqueeze(0)

        # Seleccionar y realizar un movimiento
        movimiento = seleccionar_mejor_movimiento(model, tablero)
        tablero.push(movimiento)

        # Almacenar el estado actual y la acción tomada
        historial_estados.append(estado_actual)
        historial_acciones.append(movimiento_a_indice(movimiento))

    # Calcular la recompensa final basada en el resultado del juego
    recompensa_final = calcular_recompensa_final(tablero)

    # Actualizar el modelo basado en el resultado del juego
    total_loss = 0
    for estado, accion in zip(historial_estados, historial_acciones):
        loss = calcular_perdida(model, estado, accion, recompensa_final)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss
    print("Juego:{:.0f}   Loss:{:.5f}".format(episodio+1, total_loss))
    if episodio % (num_episodios/10) == 0: 
        print(jugar_partida(model))

# %%
#################################
############# JUGAR #############
#################################

tablero = chess.Board()

# Juego interactivo en la consola
while not tablero.is_game_over():
    # Imprimir el tablero en la consola
    print(tablero)
    
    # Obtener todos los movimientos legales en la posición actual
    movimientos_legales = list(tablero.legal_moves)
    
    # Imprimir movimientos legales (opcional, para referencia)
    print("Movimientos legales:", [tablero.san(mov) for mov in movimientos_legales])
    
    # Pedir al usuario que introduzca un movimiento
    entrada_usuario = input("Introduce tu movimiento (en notación SAN): ")
    
    try:
        # Intentar realizar el movimiento
        movimiento = tablero.parse_san(entrada_usuario)
        if movimiento in movimientos_legales:
            tablero.push(movimiento)
        else:
            print("Movimiento ilegal, por favor intenta de nuevo.")
    except ValueError:
        print("Entrada inválida, por favor introduce el movimiento en notación SAN (ejemplo: e4, Nf3, ...)")
    
    computer = seleccionar_mejor_movimiento(model,tablero)
    print(computer)
    tablero.push(computer)
    # Opcional: Mostrar el tablero como SVG en Jupyter Notebook o Google Colab
    # from IPython.display import display, SVG
    # display(SVG(chess.svg.board(board=tablero)))

# Imprimir el resultado del juego
if tablero.is_checkmate():
    print("Jaque mate.")
elif tablero.is_stalemate():
    print("Tablas por ahogado.")
elif tablero.is_insufficient_material():
    print("Tablas por material insuficiente.")
elif tablero.can_claim_draw():
    print("Tablas reclamables.")

# %%
######################################
############# JUGAR SOLO #############
######################################
    
pgn = jugar_partida(model)
print(pgn)
# %%

torch.save(model.state_dict(), 'ajedres.pth')
# %%
model.load_state_dict(torch.load('ajedres.pth'))
# %%
