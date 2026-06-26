import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from main import Robot

# 1. Carregar o robô
print("Carregando modelo do robô...")
# Forçar recalculo das funções simbólicas caso não exista p_list no modelo
robot = Robot("robot.json")

# 2. Definir lista de alvos
alvo_xyz = [-700, 100, 500, 0, 0, 0]
#lista_de_alvos = [
#    [40, 0, 1300, 0, 0, 0],         # Ponto 1 (Cima)
#    [400, 400, 1000, 0, 0, 0],      # Ponto 2 (Frente Direita)
#    [400, -400, 1000, 0, 0, 0],     # Ponto 3 (Frente Esquerda)
#    [-700, 100, 500, 0, 0, 0]       # Ponto 4 (Trás Baixo)
#]
thetas_iniciais = [0, 0, 0, 0, 0, 0]

print("\nExecutando IKinQP para gerar o histórico de movimento...")
# 3. Rodar a simulação e pegar o histórico (a variável target deve ser return_history=True)
thetas_finais, history = robot.mover_para(alvo_xyz, thetas_iniciais, max_iter=300, return_history=True, modo_trajetoria='direto') # modo_trajetoria: 'direto' (padrão), 'reta', 'arco'
#thetas_finais, history = robot.mover_trajetoria(lista_de_alvos, thetas_iniciais, max_iter_por_alvo=300, return_history=True, modo_trajetoria='direto') # modo_trajetoria: 'direto' (padrão), 'reta', 'arco'

if not history:
    print("Nenhum histórico gerado.")
    exit()

print(f"\nPreparando animação com {len(history)} frames...")

# Extrair a rota completa do efetuador final (ferramenta)
path_x, path_y, path_z = [], [], []
for thetas in history:
    pts = np.array(robot.p_list_func(*thetas)).reshape(-1, 3)
    ee_pos = pts[-1]
    path_x.append(ee_pos[0])
    path_y.append(ee_pos[1])
    path_z.append(ee_pos[2])

# 4. Configurar a figura 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Inicializar o objeto linha 3D que conectará as juntas ("esqueleto")
line, = ax.plot([], [], [], 'o-', lw=3, markersize=8, color='b', label='Robô IRB 1300')
targets_scatter = ax.scatter([alvo_xyz[0]], [alvo_xyz[1]], [alvo_xyz[2]], color='r', marker='x', s=100, label='Alvo')
#targets_x = [alvo[0] for alvo in lista_de_alvos]
#targets_y = [alvo[1] for alvo in lista_de_alvos]
#targets_z = [alvo[2] for alvo in lista_de_alvos]
#targets_scatter = ax.scatter(targets_x, targets_y, targets_z, color='r', marker='x', s=100, label='Alvos')

# Desenhar o rastro (linha tracejada) da ferramenta no ar
ax.plot(path_x, path_y, path_z, '--', color='gray', alpha=0.7, lw=2, label='Trajetória Realizada')

def init():
    # Limites estáticos para o gráfico baseado no alcance típico do robô
    ax.set_xlim(-1200, 1200)
    ax.set_ylim(-1200, 1200)
    ax.set_zlim(0, 1500)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('Animação do Movimento do Robô')
    ax.legend()
    return line, targets_scatter

def update(frame):
    # Pega os ângulos da junta no frame atual
    thetas = history[frame]
    
    # Avalia a função pré-compilada p_list_func para obter as posições XYZ de todas as juntas
    pts = robot.p_list_func(*thetas)
    # Transforma a lista de vetores colunas em uma matriz N x 3 (pontos no espaço 3D)
    pts = np.array(pts).reshape(-1, 3)
    
    # Atualizar dados do esqueleto
    line.set_data(pts[:, 0], pts[:, 1])
    line.set_3d_properties(pts[:, 2])
    
    return line, targets_scatter

# 5. Configurar o layout e adicionar a barra deslizadora (Slider)

init() # Configura os limites iniciais do gráfico

plt.subplots_adjust(bottom=0.25) # Abre espaço na parte inferior
ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])

slider_frame = Slider(
    ax=ax_slider,
    label='Frame',
    valmin=0,
    valmax=len(history) - 1,
    valinit=0,
    valstep=1
)

def update_slider(val):
    frame = int(slider_frame.val)
    update(frame)
    fig.canvas.draw_idle()

slider_frame.on_changed(update_slider)

# Forçar a atualização para o frame 0 no início
update_slider(0)

plt.show()
