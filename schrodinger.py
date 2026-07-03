# =====================================================================
# PROJETO 3: SOLUÇÃO NUMÉRICA DA EQUAÇÃO DE SCHRÖDINGER DEPENDENTE DO TEMPO
# MÉTODO: CRANK-NICOLSON PARA BARREIRA QUADRADA DE POTENCIAL
# =====================================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import scipy.sparse as sparse
from scipy.sparse.linalg import splu
from IPython.display import Image as DisplayImage

try:
    import scienceplots
    plt.style.use(['science', 'notebook', 'grid'])
except ImportError:
    plt.style.use('default')

# ---------------------------------------------------------------------
# CONFIGURAÇÃO DA MALHA NUMÉRICA E PARÂMETROS FÍSICOS
# ---------------------------------------------------------------------
# Unidades adimensionais onde hbar = 1 e m = 1
Nx = 501                  # Número de pontos na malha espacial
Nt = 800                 # Número de passos de tempo totais
dx = 1.0 / (Nx - 1)       # Passo espacial
dt = 4e-6                 # Passo temporal (estável para Crank-Nicolson)
x = np.linspace(0, 1, Nx)

# Condição Inicial: Pacote de Onda Gaussiano conforme o enunciado do projeto
x0 = 0.25                 # Posição inicial do centro do pacote
sigma = 0.02              # Largura inicial do pacote (σ)
k0 = 70.0                 # Momento linear inicial (velocidade)

# Equação exata do PDF: psi(x,0) = exp(-(x-x0)^2 / (2*sigma^2)) * exp(1j*k0*x)
psi0 = np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * k0 * x)

# Condições de Contorno de Dirichlet (Partícula numa caixa: ψ = 0 nas bordas)
psi0[0] = psi0[-1] = 0

# Normalização Numérica: Garante que a integral de |ψ(x,0)|² dx = 1
psi0 /= np.sqrt(np.sum(np.abs(psi0)**2) * dx)

# Definindo a Barreira de Potencial Quadrada do Enunciado: V(x) = V0 se x em [a, b]
V0 = 4.5e3                # Altura da barreira
a, b = 0.4, 0.6         # Fronteiras espaciais da barreira
V = np.zeros(Nx)
V[(x >= a) & (x <= b)] = V0

# ---------------------------------------------------------------------
# IMPLEMENTAÇÃO DO MATRICIAL CRANK-NICOLSON (A * psi^{n+1} = B * psi^n)
# ---------------------------------------------------------------------
alpha = 1j * dt / (2 * dx**2)

# Construção da Matriz Implícita A (t + dt)
diag_A = 1.0 + 2.0 * alpha + 0.5j * dt * V
off_diag_A = -alpha * np.ones(Nx)
A_mat = sparse.diags([off_diag_A[1:], diag_A, off_diag_A[1:]], [-1, 0, 1], shape=(Nx, Nx), format='csc')

# Construção da Matriz Explícita B (t)
diag_B = 1.0 - 2.0 * alpha - 0.5j * dt * V
off_diag_B = alpha * np.ones(Nx)
B_mat = sparse.diags([off_diag_B[1:], diag_B, off_diag_B[1:]], [-1, 0, 1], shape=(Nx, Nx), format='csc')

# Decomposição LU prévia para ganho drástico de desempenho no Colab
A_solver = splu(A_mat)

# Matriz histórica para armazenar frames selecionados para os gráficos e animação
n_frames = 200
steps_per_frame = Nt // n_frames
psi_history = np.zeros((n_frames, Nx), dtype=complex)

# Loop da evolução temporal
psi = psi0.copy()
psi_history[0] = psi

frame_idx = 1
for t in range(1, Nt):
    # Calcula o vetor do lado direito: b = B * psi^n
    b_vector = B_mat.dot(psi)

    # Resolve o sistema linear tridiagonal: A * psi^{n+1} = b
    psi = A_solver.solve(b_vector)

    # Aplica rigorosamente as condições de contorno de Dirichlet
    psi[0] = psi[-1] = 0

    # Salva o frame se atingir o intervalo de amostragem
    if t % steps_per_frame == 0 and frame_idx < n_frames:
        psi_history[frame_idx] = psi
        frame_idx += 1

print("-> Solução numérica e evolução calculadas com sucesso!")

# ---------------------------------------------------------------------
# TASK 3: GRÁFICO ESTÁTICO DA DENSIDADE DE PROBABILIDADE |ψ(x, t)|²
# ---------------------------------------------------------------------
instantes = [0, int(n_frames * 0.45), n_frames - 1]
cores = ['#1f77b4', '#ff7f0e', '#2ca02c']

plt.figure(figsize=(10, 5))

# Sombreado representando geometricamente a Barreira Quadrada V(x)
V_escala = (V / V0) * (np.max(np.abs(psi_history)**2) * 0.5)
plt.fill_between(x, 0, V_escala, color='darkred', alpha=0.15, label=r'Barreira Potencial $V(x)$')
plt.axvline(a, color='darkred', linestyle=':', alpha=0.6)
plt.axvline(b, color='darkred', linestyle=':', alpha=0.6)

for idx, inst in enumerate(instantes):
    t_real = inst * steps_per_frame * dt
    plt.plot(x, np.abs(psi_history[inst])**2, color=cores[idx], linewidth=2.5,
             label=r'$|\psi|^2$ no Frame ' + f'{inst} ($t = {t_real:.3f}$ s)')

plt.xlabel('Coordenada Espacial ($x$)')
plt.ylabel(r'Densidade de Probabilidade $|\psi(x,t)|^2$')
plt.title('Instantâneos do Pacote de Onda (Antes, Durante e Depois da Colisão)')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------
# ANÁLISE DOS PARÂMETROS DO POTENCIAL: CÁLCULO NUMÉRICO DE R E T
# ---------------------------------------------------------------------
psi_final = psi_history[-1]
idx_a = np.searchsorted(x, a)
idx_b = np.searchsorted(x, b)

# Integração das sub-regiões para determinar as probabilidades de reflexão e tunelamento
R = np.sum(np.abs(psi_final[:idx_a])**2) * dx
T = np.sum(np.abs(psi_final[idx_b:])**2) * dx
Probabilidade_Total = np.sum(np.abs(psi_final)**2) * dx

print("\n" + "="*50)
print("     ANÁLISE QUANTITATIVA DOS PARÂMETROS DO POTENCIAL")
print("="*50)
print(f"Coeficiente de Reflexão (R):           {R*100:.2f}%")
print(f"Coeficiente de Transmissão/Túnel (T):  {T*100:.2f}%")
print(f"Conservação da Probabilidade Total:    {Probabilidade_Total*100:.2f}%")
print("="*50 + "\n")

# ---------------------------------------------------------------------
# VISUALIZAÇÃO DA EVOLUÇÃO (GERAÇÃO E EXIBIÇÃO DA ANIMAÇÃO EM GIF)
# ---------------------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(10, 5))

ax1.set_xlim(0, 1)
ax1.set_ylim(0, np.max(np.abs(psi_history)**2) * 1.2)
ax1.set_xlabel('Espaço ($x$)')
ax1.set_ylabel(r'Densidade de Probabilidade $|\psi(x,t)|^2$', color='blue')
line, = ax1.plot([], [], color='blue', linewidth=2.5, label=r'$|\psi(x,t)|^2$')
ax1.tick_params(axis='y', labelcolor='blue')

# Demarcação visual permanente da barreira quadrada no gráfico animado
ax1.axvspan(a, b, color='red', alpha=0.1, label='Região da Barreira [a, b]')
ax1.legend(loc='upper left')

# Eixo secundário para a escala real da energia do Potencial V(x)
ax2 = ax1.twinx()
ax2.plot(x, V, color='red', linestyle='--', alpha=0.3)
ax2.set_ylabel(r'Potencial $V(x)$', color='red')
ax2.tick_params(axis='y', labelcolor='red')

ax1.set_title('Evolução do Pacote de Onda Quântico através do Método de Crank-Nicolson')

def init():
    line.set_data([], [])
    return line,

def update(frame):
    y = np.abs(psi_history[frame])**2
    line.set_data(x, y)
    return line,

anim = FuncAnimation(fig, update, frames=n_frames, init_func=init, blit=True)

# Salvando localmente no diretório temporário do Google Colab
gif_output = 'projeto_schrodinger_final.gif'
writer = PillowWriter(fps=24)
anim.save(gif_output, writer=writer)
plt.close()

# Renderização do GIF diretamente na interface de saída do Colab
DisplayImage(open(gif_output, 'rb').read())
