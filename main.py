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
    
    def mover_para(self, x_desejado, thetas_iniciais, max_iter=200):
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
            
            if i % 10 == 0:
                print(f"Iter: {i} | Erro: {erro:.2f} | Posição: X={x_atual[0]:.1f}, Y={x_atual[1]:.1f}, Z={x_atual[2]:.1f}")
                
        return thetas
    
    def export_model(self, filename="robot_model.pkl"):
        """ Salva as matrizes simbólicas para carregamento rápido """
        with open(filename, 'wb') as f:
            data = {
                'A': self.A,
                'J': self.J,
                'theta_vars': self.theta_vars
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
            
            # Gera funções numéricas ultra-rápidas para Numpy
            self.J_func = sp.lambdify(self.theta_vars, self.J, "numpy")
            self.A_func = sp.lambdify(self.theta_vars, self.A, "numpy")
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
        if self.p_list == None:
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


if __name__ == "__main__":
    robot = Robot("robot.json")
    
    # 2. Definir o alvo cartesiano (x_desejado). Exemplo:
    # Queremos alcançar o ponto (X=665, Y=0, Z=1159) com orientação específica
    # Dica: Na primeira simulação vamos garantir que alcance a posição sem exigir orientação difícil
    alvo_xyz = [40, 0, 1300]
    alvo_rpy = [0, 0, 0] # Em casos reais, a orientação importa muito, mas para teste de alcance...
    x_desejado = alvo_xyz + alvo_rpy
    
    # 3. Posição inicial das juntas
    thetas_iniciais = [0, 0, 0, 0, 0, 0]
    
    print("\nExecutando IKinQP...")
    # 4. Rodar a simulação e resolver a cinemática inversa iterativamente
    thetas_finais = robot.mover_para(x_desejado, thetas_iniciais, max_iter=500)
    
    if thetas_finais is not None:
        print("\nJuntas Finais Encontradas (radianos):")
        print(np.round(thetas_finais, 4))
        print("Juntas Finais (graus):")
        print(np.round(np.degrees(thetas_finais), 2))