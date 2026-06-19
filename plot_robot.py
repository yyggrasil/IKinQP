import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from main import Robot

# 1. Carregar o robô
print("Carregando modelo do robô...")
# Forçar recalculo das funções simbólicas caso não exista p_list no modelo
robot = Robot("robot.json")

# 2. Definir alvo e posição inicial
alvo_xyz = [40, 0, 1300]
alvo_rpy = [0, 0, 0]
x_desejado = alvo_xyz + alvo_rpy
thetas_iniciais = [0, 0, 0, 0, 0, 0]

print("\nExecutando IKinQP para gerar o histórico de movimento...")
# 3. Rodar a simulação e pegar o histórico (a variável target deve ser return_history=True)
thetas_finais, history = robot.mover_para(x_desejado, thetas_iniciais, max_iter=500, return_history=True)

if not history:
    print("Nenhum histórico gerado.")
    exit()

print(f"\nPreparando animação com {len(history)} frames...")

# 4. Configurar a figura 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Inicializar o objeto linha 3D que conectará as juntas ("esqueleto")
line, = ax.plot([], [], [], 'o-', lw=3, markersize=8, color='b', label='Robô IRB 1300')
target_scatter = ax.scatter([alvo_xyz[0]], [alvo_xyz[1]], [alvo_xyz[2]], color='r', marker='x', s=100, label='Alvo')

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

# 5. Criar a animação
ani = animation.FuncAnimation(
    fig, update, frames=len(history), init_func=init, 
    interval=50, blit=False, repeat=False
)

plt.show()
