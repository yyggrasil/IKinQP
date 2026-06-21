import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from main import Robot

# 1. Carregar o robô
print("Carregando modelo do robô...")
# Forçar recalculo das funções simbólicas caso não exista p_list no modelo
robot = Robot("robot.json")

# 2. Definir alvo e posição inicial
alvo_xyz = [-700, 100, 500]
alvo_rpy = [0, 0, 0]
x_desejado = alvo_xyz + alvo_rpy
thetas_iniciais = [0, 0, 0, 0, 0, 0]

print("\nExecutando IKinQP para gerar o histórico de movimento...")
# 3. Rodar a simulação e pegar o histórico (a variável target deve ser return_history=True)
thetas_finais, history = robot.mover_para(x_desejado, thetas_iniciais, max_iter=500, return_history=True, modo_trajetoria='direto') # modo_trajetoria: 'direto' (padrão), 'reta', 'arco'

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
target_scatter = ax.scatter([alvo_xyz[0]], [alvo_xyz[1]], [alvo_xyz[2]], color='r', marker='x', s=100, label='Alvo')

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
    return line, target_scatter

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
    
    return line, target_scatter

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
