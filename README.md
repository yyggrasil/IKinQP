# Simulador de Cinemática de Robôs (iKinQP)

Este projeto implementa o cálculo de **Cinemática Direta (Matrizes de Transformação Homogênea)**, **Cinemática Inversa** e **Cálculo da Matriz Jacobiana** para robôs manipuladores através da parametrização de Denavit-Hartenberg (DH).

A Cinemática Inversa é resolvida de forma iterativa utilizando o inovador algoritmo de otimização **iKinQP** (Inverse Kinematics through Quadratic Programming), permitindo convergência rápida, prevenção de singularidades e suporte a futuras restrições de colisão.

## 🛠️ Tecnologias e Bibliotecas

O motor matemático híbrido aproveita o melhor dos dois mundos:
*   **SymPy**: Gera a modelagem matemática e analítica perfeita das equações da Cinemática Direta e Matriz Jacobiana.
*   **SymPy Lambdify + Pickle**: Compila as equações analíticas em funções nativas super rápidas e salva o estado no disco para carregamento quase instantâneo.
*   **NumPy e SciPy**: Utilizados para o processamento matricial e vetorial rápido.
*   **QPSolvers (OSQP)**: Solver robusto utilizado para computar a otimização quadrática da cinemática inversa iterativa em tempo real.

## 📦 Como Instalar

Certifique-se de ter o Python 3.10+ instalado. Em seguida, ative seu ambiente virtual e instale as dependências executando o comando abaixo na pasta do projeto:

```bash
pip install numpy scipy qpsolvers osqp sympy
```

## 🚀 Como Usar

### 1. Configurando seu Robô
O modelo mecânico do robô é lido do arquivo `robot.json`. Basta preencher os parâmetros DH de cada junta do seu manipulador. 
Exemplo de uma junta no arquivo JSON:
```json
{
    "type": "revolute",
    "a": 150,
    "alpha": -90,
    "d": 544,
    "theta": "theta1",
    "offset": 0,
    "limits": [-180, 180]
}
```

### 2. Rodando o Script
Execute o script principal:
```bash
python main.py
```

* **Na primeira execução**, o programa pode levar alguns segundos gerando as fórmulas algébricas exatas e compilando as funções. Um arquivo `robot_model.pkl` será salvo na pasta.
* **Nas próximas execuções**, o robô é carregado quase instantaneamente pelo cache serializado.

### 3. Modificando o Alvo
No final do arquivo `main.py`, você pode alterar a variável `alvo_xyz` para a coordenada cartesiana X, Y e Z que deseja que a ponta da ferramenta alcance.
```python
alvo_xyz = [40, 0, 1300]   # Digite sua posição em milímetros aqui
alvo_rpy = [0, 0, 0]       # Orientação em Euler Roll, Pitch, Yaw
x_desejado = alvo_xyz + alvo_rpy
```
O robô iterará pelo solver OSQP e imprimirá os ângulos necessários de cada junta para atingir a referida posição.

## 🗂️ Arquivos do Projeto

*   `main.py`: Código principal com a classe `Robot` que analisa os parâmetros DH, calcula a Jacobiana iterativa e possui o loop solver.
*   `ikinqp_solver.py`: Arquivo de matemática que compõe a formulação H (Hessiana) e g (Gradiente) para o Problema de Otimização Quadrática.
*   `robot.json`: Arquivo de configuração das juntas físicas.
*   `robot_model.pkl`: Arquivo auto-gerado que armazena a compilação simbólica para velocidade extra.
