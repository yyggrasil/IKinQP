import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import sparse
from qpsolvers import solve_qp
from main import Robot
from ikinqp_solver import IKinQPSympy

# 1. Carregar robôs
print("Carregando modelos dos robôs...")
robot1 = Robot("robot.json")
robot2 = Robot("robot.json")

# Offsets para as bases (separados por 800 mm no eixo Y)
offset_base_1 = np.array([0.0, 0.0, 0.0])
offset_base_2 = np.array([0.0, 800.0, 0.0])

# Posições iniciais das juntas
thetas1 = np.array([0, 0, 0, 0, 0, 0], dtype=float)
thetas2 = np.array([0, 0, 0, 0, 0, 0], dtype=float)

# Alvos (Ambos tentando pegar o mesmo "objeto" na posição [500, 400, 1000])
# Isso forçará os robôs a se aproximarem muito e ativar a prevenção de colisão
alvo_1 = np.array([500.0, 400.0, 1000.0, 0.0, 0.0, 0.0])
alvo_2 = np.array([500.0, 350.0, 1000.0, 0.0, 0.0, 0.0]) 

# Configurações do solver e simulação
max_iter = 400
dt = 0.05
gamma_gain = np.eye(6) * 150.0
lam = 0.01

history1 = [thetas1.copy()]
history2 = [thetas2.copy()]
history_p1 = []
history_p2 = []

ikin1 = IKinQPSympy(n_dof=6, n_task=6)
ikin2 = IKinQPSympy(n_dof=6, n_task=6)

# Margem de segurança de 250 mm entre os efetuadores (impede colisão das garras)
d_min = 100.0 
# Ganho do Controlador de Barreira (CBF) para evasão de colisão
gamma_c = 15.0 

print("Iniciando simulação multi-robô com prevenção de colisão...")
for i in range(max_iter):
    # --- Estado do Robô 1 ---
    x_atual_1 = robot1.get_pose(thetas1)
    x_atual_1[0:3] += offset_base_1
    J1 = robot1.J_func(*thetas1)
    
    # --- Estado do Robô 2 ---
    x_atual_2 = robot2.get_pose(thetas2)
    x_atual_2[0:3] += offset_base_2
    J2 = robot2.J_func(*thetas2)
    
    # Rastrear posição do efetuador no histórico
    p1 = x_atual_1[0:3]
    p2 = x_atual_2[0:3]
    history_p1.append(p1.copy())
    history_p2.append(p2.copy())
    
    # --- Avaliar matrizes QP individuais (H_eval tem size 6x6, g_eval size 6) ---
    H1, g1 = ikin1.evaluate_qp_matrices(J1, x_atual_1, alvo_1, np.zeros(6), gamma_gain, lam, dt)
    H2, g2 = ikin2.evaluate_qp_matrices(J2, x_atual_2, alvo_2, np.zeros(6), gamma_gain, lam, dt)
    
    # --- Montar QP Conjunto (12 variáveis: q_dot_1 e q_dot_2) ---
    H = sparse.block_diag((H1, H2), format='csc')
    g = np.concatenate((g1, g2))
    
    # --- Restrição de Prevenção de Colisão para TODAS as juntas ---
    G_list = []
    h_list = []
    
    # Extrair listas de posições e eixos Z para todas as juntas
    pts1 = np.array(robot1.p_list_func(*thetas1)).reshape(-1, 3) + offset_base_1
    zs1 = np.array(robot1.z_list_func(*thetas1)).reshape(-1, 3)
    
    pts2 = np.array(robot2.p_list_func(*thetas2)).reshape(-1, 3) + offset_base_2
    zs2 = np.array(robot2.z_list_func(*thetas2)).reshape(-1, 3)
    
    min_dist_atual = float('inf')
    
    # Iterar sobre as juntas do braço (índices 1 a 6)
    for j in range(1, 7):
        p1_j = pts1[j]
        # Jacobiano Jv1_j (3x6) da junta j do robô 1
        Jv1_j = np.zeros((3, 6))
        for k_idx in range(j):
            Jv1_j[:, k_idx] = np.cross(zs1[k_idx], p1_j - pts1[k_idx])
            
        for k in range(1, 7):
            p2_k = pts2[k]
            # Jacobiano Jv2_k (3x6) da junta k do robô 2
            Jv2_k = np.zeros((3, 6))
            for k_idx in range(k):
                Jv2_k[:, k_idx] = np.cross(zs2[k_idx], p2_k - pts2[k_idx])
                
            # Distância entre as juntas j (R1) e k (R2)
            diff = p1_j - p2_k
            d = np.linalg.norm(diff)
            
            if d < min_dist_atual:
                min_dist_atual = d
            
            if d < 1e-3: 
                diff = np.array([0, 1e-3, 0])
                d = 1e-3
                
            n = diff / d # vetor unitário de 2 para 1
            
            # Equação da Barreira: -(n^T * Jv1_j) * q1_dot + (n^T * Jv2_k) * q2_dot <= gamma_c * (d - d_min)
            G1_jk = -n.T @ Jv1_j # 1x6
            G2_jk = n.T @ Jv2_k  # 1x6
            
            G_list.append(np.concatenate((G1_jk, G2_jk)))
            h_list.append(gamma_c * (d - d_min))
            
    # Matriz G e vetor h para todas as restrições
    G = sparse.csc_matrix(np.vstack(G_list))
    h = np.array(h_list)
    
    # Limites físicos de velocidade
    lb = np.ones(12) * -15.0
    ub = np.ones(12) * 15.0
    
    try:
        # Tenta resolver o QP usando OSQP
        q_dot = solve_qp(H, g, G=G, h=h, lb=lb, ub=ub, solver='osqp')
        if q_dot is None:
            # Fallback para outro solver (ex: clarabel) caso OSQP falhe numéricamente
            q_dot = solve_qp(H, g, G=G, h=h, lb=lb, ub=ub, solver='clarabel')
            if q_dot is None:
                print("QP Inviável (Local Minima), parando movimento.")
                q_dot = np.zeros(12)
                break
    except Exception as e:
        print(f"Erro no solver: {e}")
        break
        
    q_dot_1 = q_dot[0:6]
    q_dot_2 = q_dot[6:12]
    
    # Atualizar posições (Euler integration)
    thetas1 = thetas1 + q_dot_1 * dt
    thetas2 = thetas2 + q_dot_2 * dt
    
    history1.append(thetas1.copy())
    history2.append(thetas2.copy())
    
    erro1 = np.linalg.norm(alvo_1[0:3] - p1)
    erro2 = np.linalg.norm(alvo_2[0:3] - p2)
    if i % 20 == 0:
        print(f"Iter {i:3d} | Erro R1: {erro1:5.1f} | Erro R2: {erro2:5.1f} | Dist Mínima entre Robôs: {min_dist_atual:5.1f}")
        
    if erro1 < 2.0 and erro2 < d_min + 2.0:
        # Quando um atinge e o outro não consegue mais se aproximar
        pass

print(f"\nSimulação concluída! Preparando animação com {len(history_p1)} frames...")

# --- Gráficos e Animação ---
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Esqueletos
line1, = ax.plot([], [], [], 'o-', lw=3, markersize=8, color='blue', label='Robô 1')
line2, = ax.plot([], [], [], 'o-', lw=3, markersize=8, color='green', label='Robô 2')

# Trajetórias do efetuador final
traj1, = ax.plot([], [], [], '--', color='cyan', lw=2, label='Trajetória R1')
traj2, = ax.plot([], [], [], '--', color='lime', lw=2, label='Trajetória R2')

# Marcador do Alvo
ax.scatter([alvo_1[0]], [alvo_1[1]], [alvo_1[2]], color='red', marker='x', s=150, label='Alvo de Ambos', lw=3)

def init():
    # Visão estática do ambiente
    ax.set_xlim(-800, 1200)
    ax.set_ylim(-200, 1200)
    ax.set_zlim(0, 1500)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('Colaboração Multi-Robô com Prevenção de Colisões (iKinQP)')
    ax.legend()
    return line1, line2, traj1, traj2

def update(frame):
    # Robô 1
    t1 = history1[frame]
    pts1 = np.array(robot1.p_list_func(*t1)).reshape(-1, 3)
    pts1 += offset_base_1
    line1.set_data(pts1[:, 0], pts1[:, 1])
    line1.set_3d_properties(pts1[:, 2])
    
    # Robô 2
    t2 = history2[frame]
    pts2 = np.array(robot2.p_list_func(*t2)).reshape(-1, 3)
    pts2 += offset_base_2
    line2.set_data(pts2[:, 0], pts2[:, 1])
    line2.set_3d_properties(pts2[:, 2])
    
    # Rastro / Trajetória R1
    h_p1 = np.array(history_p1[:max(1, frame)])
    traj1.set_data(h_p1[:, 0], h_p1[:, 1])
    traj1.set_3d_properties(h_p1[:, 2])
    
    # Rastro / Trajetória R2
    h_p2 = np.array(history_p2[:max(1, frame)])
    traj2.set_data(h_p2[:, 0], h_p2[:, 1])
    traj2.set_3d_properties(h_p2[:, 2])
    
    return line1, line2, traj1, traj2

ani = animation.FuncAnimation(
    fig, update, frames=len(history_p1), init_func=init, 
    interval=50, blit=False, repeat=False
)

plt.show()
