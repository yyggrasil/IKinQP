import os
import json
import pickle
import sympy as sp
import numpy as np
from scipy import sparse
from qpsolvers import solve_qp
from ikinqp_solver import IKinQPSympy

# CONVERTE VALOR DO JSON

# LER OS VALORES DO JSON
class Robot:
    def __init__(self, filename, robotModel="robot_model.pkl"):
        """
        Construtor da Classe
        """
        # Valores padrões
        self.A = None
        robot_data = None
        self.z_list = None
        self.p_list = None

        try:
            with open(filename, 'r', encoding='utf-8') as f:
                robot_data = json.load(f)
        except Exception as e:
            print(f"Erro ao ler o arquivo JSON: {e}")
            return

        self.joints = []
        self.theta_vars = []
        
        # Extrair os parâmetros DH do JSON
        for joint_data in robot_data.get('joints', []):
            j_type = joint_data.get('type', 'revolute')
            
            # Chaves de acordo com o padrão DH (theta, d, a, alpha)
            theta_str = joint_data.get('theta', '0')
            d_str = joint_data.get('d', '0')
            a_str = joint_data.get('a', '0')
            alpha_str = joint_data.get('alpha', '0')
            offset_str = joint_data.get('offset', '0')

            #limitação fisica do robô
            limits = joint_data.get('limits', [])
            lim_min = sp.rad(limits[0])
            lim_max = sp.rad(limits[1])
            
            parsed_theta = self.parse_value(theta_str, is_angle=True)
            if hasattr(parsed_theta, 'free_symbols') and parsed_theta.free_symbols:
                self.theta_vars.append(parsed_theta)
                
            self.joints.append({
                'type': j_type,
                'theta': parsed_theta,
                'd': self.parse_value(d_str),
                'a': self.parse_value(a_str),
                'alpha': self.parse_value(alpha_str, is_angle=True),
                'offset': self.parse_value(offset_str, is_angle=True),
                'limits': [lim_min, lim_max]
            })
        self.n_joints = len(self.joints)

        # tenta Carregar os valores das matrizes (calculo mais rápido)
        if not self.load_model(robotModel):
            print('='*5)
            print("calculando a matriz Jacobiana pela primeira vez, pode demorar um pouco...")
            self.calc_J()
    
    def get_matrix(self, theta, d, a, alpha):
        """
        Retorna a matriz de transformação homogênea padrão usando o modelo Denavit-Hartenberg.
        """
        A = sp.Matrix([
            [sp.cos(theta), -sp.cos(alpha)*sp.sin(theta),  sp.sin(alpha)*sp.sin(theta), a*sp.cos(theta)],
            [sp.sin(theta),  sp.cos(alpha)*sp.cos(theta), -sp.sin(alpha)*sp.cos(theta), a*sp.sin(theta)],
            [0,              sp.sin(alpha),                sp.cos(alpha),               d              ],
            [0,              0,                            0,                           1              ]
        ])
        return sp.simplify(A)
    
    def parse_value(self, val, is_angle=False):
        """
        Converte os valores do json
        """
        if val is None or str(val).strip() == '':
            return 0
        try:
            parsed = sp.sympify(str(val))
        except Exception:
            parsed = sp.Symbol(str(val))
            
        if is_angle:
            if hasattr(parsed, 'free_symbols') and not parsed.free_symbols and not parsed.has(sp.pi) or isinstance(parsed, (int, float)):
                return sp.rad(parsed)
                
        return parsed
    
    def calc_A(self):
        """
        Calcula a matriz de transformação final do robô
        """
        # A Matriz de Transformação da Base (Identidade no inicio)
        self.A = sp.eye(4)
        
        # Vetores z_i e posições p_i
        self.z_list = [sp.Matrix([0, 0, 1])]
        self.p_list = [sp.Matrix([0, 0, 0])]
        for j in self.joints:
            A_i = self.get_matrix(
                theta = j['theta'] + j['offset'],
                d =     j['d'],
                a =     j['a'],
                alpha = j['alpha']
            )
            self.A = self.A * A_i   
            self.A = sp.simplify(self.A)

            # Extrair z_i (eixo z) para a Jacobiana
            z_i = self.A[0:3, 2] 
            self.z_list.append(z_i)
            
            # Extrair p_i (posição) para a Jacobiana
            p_i = self.A[0:3, 3]
            self.p_list.append(p_i)
    
    def get_pose(self, thetas_num):
        """ Retorna vetor x (6x1) com [X, Y, Z, Roll, Pitch, Yaw] """        
        import numpy as np
        
        A_num = self.A_func(*thetas_num)
        
        X = A_num[0, 3]
        Y = A_num[1, 3]
        Z = A_num[2, 3]

        sy = np.sqrt(A_num[0,0] * A_num[0,0] + A_num[1,0] * A_num[1,0])
        singular = sy < 1e-6
        if not singular:
            roll = np.arctan2(A_num[2,1], A_num[2,2])
            pitch = np.arctan2(-A_num[2,0], sy)
            yaw = np.arctan2(A_num[1,0], A_num[0,0])
        else:
            roll = np.arctan2(-A_num[1,2], A_num[1,1])
            pitch = np.arctan2(-A_num[2,0], sy)
            yaw = 0
            
        return np.array([X, Y, Z, roll, pitch, yaw], dtype=float)
    
    def mover_para(self, x_desejado, thetas_iniciais, max_iter=200, return_history=False):
        """ Move o robô usando o algoritmo iterativo iKinQP. """
        
        if not hasattr(self, 'J_func'):
            self.calc_A()
            self.calc_J()
            
        ikin = IKinQPSympy(n_dof=self.n_joints, n_task=6)
        
        thetas = np.array(thetas_iniciais, dtype=float)
        x_d = np.array(x_desejado, dtype=float)
        x_dot_d = np.zeros(6) # Queremos parar no alvo
        
        gamma = np.eye(6) * 150.0 # Ganho de convergência
        lam = 0.01 # Amortecimento de singularidade
        dt = 0.05 # Passo de simulação
        
        print(f"Iniciando movimento para alvo:\n{x_d}")
        
        history = [thetas.copy()] if return_history else None
        
        for i in range(max_iter):
            # 1. Obter Pose atual e Jacobiano numericamente
            x_atual = self.get_pose(thetas)
            J_atual = self.J_func(*thetas)
            
            # 2. Calcular erro
            erro = np.linalg.norm(x_d - x_atual)
            if erro < 1.0: # Tolerância (1mm de posição / 1rad de giro total)
                print(f"\nAlvo alcançado na iteração {i}! Erro final: {erro:.3f}")
                break
                
            # 3. Montar matrizes QP
            H, g = ikin.evaluate_qp_matrices(J_atual, x_atual, x_d, x_dot_d, gamma, lam, dt)
            
            # Converter H para matriz esparsa CSC para máxima performance no OSQP (remove aviso)
            
            H_sparse = sparse.csc_matrix(H)
            
            # 4. Resolver QP (qpsolvers osqp)
            try:
                # lb e ub para limites de junta poderiam ser adicionados aqui
                q_dot = solve_qp(H_sparse, g, solver='osqp')
                if q_dot is None:
                    print("\nSolver falhou. Abortando.")
                    break
            except Exception as e:
                print(f"\nErro no solver: {e}")
                break
                
            # 5. Atualizar juntas
            thetas = thetas + q_dot * dt
            
            if return_history:
                history.append(thetas.copy())
            
            if i % 10 == 0:
                print(f"Iter: {i} | Erro: {erro:.2f} | Posição: X={x_atual[0]:.1f}, Y={x_atual[1]:.1f}, Z={x_atual[2]:.1f}")
                
        if return_history:
            return thetas, history
        return thetas
    
    def export_model(self, filename="robot_model.pkl"):
        """ Salva as matrizes simbólicas para carregamento rápido """
        with open(filename, 'wb') as f:
            data = {
                'A': self.A,
                'J': self.J,
                'theta_vars': self.theta_vars,
                'p_list': self.p_list,
                'z_list': self.z_list
            }
            pickle.dump(data, f)
        print(f"Modelo exportado para {filename} com sucesso!")
    
    def load_model(self, filename="robot_model.pkl"):
        """ Carrega as matrizes simbólicas e gera as funções numéricas """
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.A = data['A']
                self.J = data['J']
                self.theta_vars = data['theta_vars']
                self.p_list = data.get('p_list', None)
                self.z_list = data.get('z_list', None)
            
            if self.p_list is None or self.z_list is None:
                return False # Força o recalculo
            
            # Gera funções numéricas ultra-rápidas para Numpy
            self.J_func = sp.lambdify(self.theta_vars, self.J, "numpy")
            self.A_func = sp.lambdify(self.theta_vars, self.A, "numpy")
            self.p_list_func = sp.lambdify(self.theta_vars, self.p_list, "numpy")
            self.z_list_func = sp.lambdify(self.theta_vars, self.z_list, "numpy")
            print(f"Modelo '{filename}' carregado com sucesso (cálculo instantâneo)!")
            return True
        return False
    
    def calc_mat_final(self, vals):
        """
        Substitui o valor da matriz de transformação
        """
        vals = {
            chave: sp.rad(valor) if chave.startswith('theta') else valor
            for chave, valor in vals.items()    
        }
        if self.A == None:
            self.calc_A()
        sp.pprint(self.A.subs(vals).evalf())
    
    def calc_J(self, saveFile="robot_model.pkl"):
        """
        Calculo da matriz jacobiana do Robô
        """
        if self.p_list is None or self.z_list is None:
            self.calc_A()

        self.J = sp.zeros(6, self.n_joints)
        p_n = self.p_list[-1] # Posição final (Efetuador)

        for i in range(self.n_joints):
            j_type = self.joints[i]['type'].lower()
        
            z_prev = self.z_list[i]
            p_prev = self.p_list[i]
            
            if j_type == 'revolute' or j_type == 'rotational':
                p_diff = p_n - p_prev
                Jv = z_prev.cross(p_diff)
                Jw = z_prev
            elif j_type == 'prismatic':
                Jv = z_prev
                Jw = sp.zeros(3,1)
            else:
                print(f"AVISO: O tipo de junta '{j_type}' é desconhecido. Assumindo que é revoluta.")
                p_diff = p_n - p_prev
                Jv = z_prev.cross(p_diff)
                Jw = z_prev
                
            self.J[0:3, i] = Jv
            self.J[3:6, i] = Jw
        self.J = sp.simplify(self.J)
        self.export_model(saveFile)
        
        # Gera funções numéricas ultra-rápidas para Numpy
        self.J_func = sp.lambdify(self.theta_vars, self.J, "numpy")
        self.A_func = sp.lambdify(self.theta_vars, self.A, "numpy")
        self.p_list_func = sp.lambdify(self.theta_vars, self.p_list, "numpy")
        self.z_list_func = sp.lambdify(self.theta_vars, self.z_list, "numpy")

        self.z_list_func = sp.lambdify(self.theta_vars, self.z_list, "numpy")


def closest_points_on_segments(p1, p2, p3, p4):
    """ Calcula os pontos mais próximos entre o segmento p1->p2 e o segmento p3->p4 """
    EPS = 1e-6
    d1 = p2 - p1
    d2 = p4 - p3
    r = p1 - p3
    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)
    
    if a <= EPS and e <= EPS:
        return p1, p3
    if a <= EPS:
        s = 0.0
        t = f / e
        t = np.clip(t, 0.0, 1.0)
    else:
        c = np.dot(d1, r)
        if e <= EPS:
            t = 0.0
            s = np.clip(-c / a, 0.0, 1.0)
        else:
            b = np.dot(d1, d2)
            denom = a * e - b * b
            if denom != 0.0:
                s = np.clip((b * f - c * e) / denom, 0.0, 1.0)
            else:
                s = 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0
                s = np.clip((b - c) / a, 0.0, 1.0)
                
    cp1 = p1 + d1 * s
    cp2 = p3 + d2 * t
    return cp1, cp2

class MultiRobotSystem:
    def __init__(self, robots_list, offsets_list):
        self.robots = robots_list
        self.offsets = [np.array(off, dtype=float) for off in offsets_list]
        self.N = len(self.robots)
        
    def calc_min_dist_pair(self, idx1, thetas_list, idx2):
        t1 = thetas_list[idx1]
        t2 = thetas_list[idx2]
        pts1 = np.array(self.robots[idx1].p_list_func(*t1)).reshape(-1, 3) + self.offsets[idx1]
        pts2 = np.array(self.robots[idx2].p_list_func(*t2)).reshape(-1, 3) + self.offsets[idx2]
        
        min_dist = float('inf')
        for j in range(1, 7):
            p1_start = pts1[j-1]
            p1_end = pts1[j]
            for k in range(1, 7):
                p2_start = pts2[k-1]
                p2_end = pts2[k]
                cp1, cp2 = closest_points_on_segments(p1_start, p1_end, p2_start, p2_end)
                d = np.linalg.norm(cp1 - cp2)
                if d < min_dist:
                    min_dist = d
        return min_dist

    def mover_trajetoria_conjunta(self, alvos_dict, thetas_iniciais_dict, max_iter=400, plotar=True):
        from qpsolvers import solve_qp
        from scipy import sparse
        from ikinqp_solver import IKinQPSympy
        import matplotlib.pyplot as plt

        d_min = 250.0
        gamma_c = 15.0
        dt = 0.05
        lam = 0.01
        gamma_gain = np.eye(6) * 150.0
        
        thetas_list = [np.array(thetas_iniciais_dict[i], dtype=float) for i in range(self.N)]
        ikins = [IKinQPSympy(n_dof=6, n_task=6) for _ in range(self.N)]
        
        if plotar:
            plt.ion()
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            colors = ['blue', 'green', 'orange', 'purple', 'cyan']
            lines = []
            targets = []
            for i in range(self.N):
                c = colors[i % len(colors)]
                line, = ax.plot([], [], [], 'o-', lw=3, markersize=6, color=c, label=f"Robô {i+1}")
                lines.append(line)
                tgt, = ax.plot([], [], [], '*', color=c, markersize=12, label=f"Alvo {i+1}")
                targets.append(tgt)
                
            ax.set_xlim(-1500, 2500)
            ax.set_ylim(-1500, 2500)
            ax.set_zlim(0, 2000)
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            ax.set_zlabel('Z (mm)')
            ax.legend()
            plt.title("Multi-Robô Cinemática Inversa iterativa (iKinQP)")
            
        print("Iniciando simulação Multi-Robô...")
        for i in range(self.N):
            if plotar:
                xd = alvos_dict[i]
                targets[i].set_data([xd[0]], [xd[1]])
                targets[i].set_3d_properties([xd[2]])
                
        for i_iter in range(max_iter):
            H_blocks = []
            g_list = []
            
            # Matrizes base de rastreamento para todos os robôs
            for i in range(self.N):
                x_atual = self.robots[i].get_pose(thetas_list[i])
                x_atual[0:3] += self.offsets[i]
                J = self.robots[i].J_func(*thetas_list[i])
                
                H_i, g_i = ikins[i].evaluate_qp_matrices(J, x_atual, alvos_dict[i], np.zeros(6), gamma_gain, lam, dt)
                H_blocks.append(H_i)
                g_list.append(g_i)
                
                if plotar and i_iter % 2 == 0:
                    pts = np.array(self.robots[i].p_list_func(*thetas_list[i])).reshape(-1, 3) + self.offsets[i]
                    lines[i].set_data(pts[:, 0], pts[:, 1])
                    lines[i].set_3d_properties(pts[:, 2])
            
            if plotar and i_iter % 2 == 0:
                plt.pause(0.01)
                
            H = sparse.block_diag(H_blocks, format='csc')
            g = np.concatenate(g_list)
            
            G_col_list = []
            h_col_list = []
            
            # Colisão N-para-N (Gradiente Numérico iKinQP - Eq 13)
            eps = 1e-4
            for a in range(self.N):
                for b in range(a + 1, self.N):
                    dist_ab = self.calc_min_dist_pair(a, thetas_list, b)
                    if dist_ab < d_min + 150.0:
                        grad_d = np.zeros(self.N * 6)
                        
                        # Perturbações no Robô A
                        for m in range(6):
                            tl_plus = [t.copy() for t in thetas_list]
                            tl_plus[a][m] += eps
                            d_plus = self.calc_min_dist_pair(a, tl_plus, b)
                            
                            tl_minus = [t.copy() for t in thetas_list]
                            tl_minus[a][m] -= eps
                            d_minus = self.calc_min_dist_pair(a, tl_minus, b)
                            
                            grad_d[a*6 + m] = (d_plus - d_minus) / (2 * eps)
                            
                        # Perturbações no Robô B
                        for m in range(6):
                            tl_plus = [t.copy() for t in thetas_list]
                            tl_plus[b][m] += eps
                            d_plus = self.calc_min_dist_pair(a, tl_plus, b)
                            
                            tl_minus = [t.copy() for t in thetas_list]
                            tl_minus[b][m] -= eps
                            d_minus = self.calc_min_dist_pair(a, tl_minus, b)
                            
                            grad_d[b*6 + m] = (d_plus - d_minus) / (2 * eps)
                            
                        # Adicionar restrição da colisão entre A e B
                        G_col_list.append(-grad_d)
                        h_col_list.append(gamma_c * (dist_ab - d_min))
                        
            if len(G_col_list) > 0:
                G = sparse.csc_matrix(np.vstack(G_col_list))
                h = np.array(h_col_list)
            else:
                G = None
                h = None
                
            lb = np.ones(self.N * 6) * -15.0
            ub = np.ones(self.N * 6) * 15.0
            
            try:
                if G is not None:
                    q_dot = solve_qp(H, g, G=G, h=h, lb=lb, ub=ub, solver='osqp')
                else:
                    q_dot = solve_qp(H, g, lb=lb, ub=ub, solver='osqp')
                    
                if q_dot is None:
                    if G is not None:
                        q_dot = solve_qp(H, g, G=G, h=h, lb=lb, ub=ub, solver='clarabel')
                    else:
                        q_dot = solve_qp(H, g, lb=lb, ub=ub, solver='clarabel')
                        
                    if q_dot is None:
                        print("QP Inviável (Local Minima)")
                        q_dot = np.zeros(self.N * 6)
                        break
            except Exception as e:
                print(f"Erro no solver: {e}")
                break
                
            for i in range(self.N):
                thetas_list[i] += q_dot[i*6 : (i+1)*6] * dt
                
        print("Trajetória Multi-Robô concluída!")
        if plotar:
            plt.ioff()
            plt.show()
        return thetas_list


if __name__ == "__main__":
    robot1 = Robot("robot.json")
    robot2 = Robot("robot.json")
    
    sistema = MultiRobotSystem(
        robots_list=[robot1, robot2],
        offsets_list=[[0, 0, 0], [0, 800, 0]]
    )
    
    # Ambos os robôs tentam pegar um alvo conflitante na mesma posição!
    alvos_dict = {
        0: [500, 400, 1000, 0, 0, 0],
        1: [500, 200, 1000, 0, 0, 0]
    }
    
    iniciais_dict = {
        0: [0, 0, 0, 0, 0, 0],
        1: [0, 0, 0, 0, 0, 0]
    }
    
    print("\nExecutando Cena Multi-Robô com Evasão de Colisão iKinQP...")
    thetas_finais = sistema.mover_trajetoria_conjunta(alvos_dict, iniciais_dict, max_iter=200, plotar=True)