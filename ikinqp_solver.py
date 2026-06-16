import sympy as sp
import numpy as np

class IKinQPSympy:
    """
    Implementação simbólica do algoritmo iKinQP (Collision-Free Inverse Kinematics 
    Through QP Optimization) utilizando a biblioteca SymPy.
    
    Baseado nas equações do artigo científico fornecido.
    """
    def __init__(self, n_dof=7, n_task=6):
        self.n_dof = n_dof
        self.n_task = n_task
        
        # --- Variáveis Escalares ---
        self.dt = sp.Symbol('dt', real=True, positive=True)       # Passo de tempo (delta t)
        self.lam = sp.Symbol('lambda', real=True, positive=True)  # Peso de penalização da velocidade articular
        self.d_buff = sp.Symbol('d_buff', real=True)              # Distância de buffer de segurança (margem para colisão)
        
        # --- Matrizes e Vetores Simbólicos ---
        # Matriz Jacobiana (fornecida pelo usuário)
        self.J = sp.MatrixSymbol('J', n_task, n_dof)
        
        # Matriz Gamma (pesos para compensar o drift)
        self.gamma = sp.MatrixSymbol('gamma', n_task, n_task)
        
        # Posições e Velocidades
        self.q = sp.MatrixSymbol('q', n_dof, 1)                   # Posição articular atual
        self.x = sp.MatrixSymbol('x', n_task, 1)                  # Pose cartesiana atual (extraída da Matriz de Transformação)
        self.x_d = sp.MatrixSymbol('x_d', n_task, 1)              # Pose cartesiana desejada
        self.x_dot_d = sp.MatrixSymbol('x_dot_d', n_task, 1)      # Velocidade cartesiana desejada
        
        # Matriz Identidade
        self.I = sp.Identity(n_dof)
        
        # --- Formulação do Problema QP (Equação 8 do artigo) ---
        # min (1/2) * q_dot^T * H * q_dot + q_dot^T * g
        
        # Matriz Hessiana H
        self.H = self.J.T * self.J + (self.dt**2) * self.J.T * self.gamma * self.J + self.lam * self.I
        
        # Vetor Gradiente g
        self.g = -self.J.T * self.x_dot_d + self.dt * self.J.T * self.gamma * (self.x - self.x_d)
        
        # --- Formulação de Restrições de Colisão (Equação 12 do artigo) ---
        # Gradiente da distância em relação a q (Jacobiano da colisão)
        self.ddist_dq = sp.MatrixSymbol('ddist_dq', 1, n_dof) 
        
        # Matriz de desigualdade de restrição A
        self.A = self.ddist_dq * self.dt
        
        # Limites da desigualdade (lbA <= A * q_dot <= ubA)
        self.lbA = sp.Matrix([self.d_buff])
        self.ubA = sp.Matrix([sp.oo]) # Representa um limite superior infinito

    def evaluate_qp_matrices(self, J_val, x_val, x_d_val, x_dot_d_val, gamma_val, lam_val, dt_val):
        """
        Avalia as equações utilizando operações vetorizadas em Numpy para o solver iterativo numérico.
        
        Returns:
            H_eval, g_eval: Matrizes no formato Numpy prontas para o qpsolvers.
        """
        J_np = np.array(J_val, dtype=float)
        gamma_np = np.array(gamma_val, dtype=float)
        x_np = np.array(x_val, dtype=float).reshape(-1, 1)
        x_d_np = np.array(x_d_val, dtype=float).reshape(-1, 1)
        x_dot_d_np = np.array(x_dot_d_val, dtype=float).reshape(-1, 1)
        I_np = np.eye(self.n_dof)
        
        H_eval = J_np.T @ J_np + (dt_val**2) * (J_np.T @ gamma_np @ J_np) + lam_val * I_np
        g_eval = -J_np.T @ x_dot_d_np + dt_val * (J_np.T @ gamma_np @ (x_np - x_d_np))
        
        # qpsolvers exige que H seja estritamente simétrica (evita erros numéricos)
        H_eval = (H_eval + H_eval.T) / 2.0
        
        return H_eval, g_eval.flatten()

if __name__ == "__main__":
    # Inicializando o modelo simbólico (ex: 7 graus de liberdade, 6 dimensões no espaço da tarefa)
    ikinqp = IKinQPSympy(n_dof=7, n_task=6)
    
    print("="*50)
    print(" FORMULAÇÃO DO PROBLEMA IKINQP (SymPy)")
    print("="*50)
    
    print("\n[1] MATRIZ HESSIANA (H) PARA O SOLVER QP:")
    sp.pprint(ikinqp.H)
    
    print("\n[2] VETOR GRADIENTE (g) PARA O SOLVER QP:")
    sp.pprint(ikinqp.g)
    
    print("\n[3] MATRIZ DE RESTRIÇÃO DE COLISÃO (A):")
    sp.pprint(ikinqp.A)
    
    print("\n[!] DICA: Para usar, extraia o vetor de pose 'x_val' da sua Matriz de Transformação Homogênea,")
    print("          e passe juntamente com a Matriz Jacobiana 'J_val' para a função 'evaluate_qp_matrices'.")
 