## cod principal
# funciona bem as splines


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Union
import numpy.typing as npt

def main():
    # Carrega os dados do arquivo Excel
    data = pd.ExcelFile('super_tanker.xlsx')
    
    # Lê a planilha BxWL (sheet_name=1 ou 'BxWL')
    ct = pd.read_excel(data, sheet_name='BxWLxDunkerque', header=None)
    
    # BALIZAS - primeira coluna (ignorando o cabeçalho)
    x = ct.iloc[1:, 0].values.astype(float).flatten()
    
    # WL (Waterlines) - primeira linha (ignorando a primeira célula)
    wl = ct.iloc[0, 1:].values.astype(float).flatten()
    
    # Solicita input do usuário com validação
    while True:
        try:
            splines = int(input("Digite o valor de pontos entre balizas desejado (0 para usar os pontos originais): "))
            if splines >= 0:
                break
            print("Por favor, digite um número não negativo.")
        except ValueError:
            print("Por favor, digite um número inteiro válido.")
    
    # Gera pontos interpolados
    x_spl = np.array(spline_x(x, splines))
    z_spl = np.array(spline_z(wl, splines))
    
    # Gera os gráficos
    grafico_X_Y(x, wl, ct, x_spl)
    grafico_Z_Y(x, wl, ct, z_spl)

def spline_x(x: npt.NDArray[np.float64], splines: int) -> npt.NDArray[np.float64]:
    """Interpola os pontos de X (balizas) conforme a densidade especificada."""
    if splines == 0:
        return x.copy()
    else:
        step = (x[1] - x[0]) / splines
        return np.arange(x[0], x[-1] + step, step)

def spline_z(wl: npt.NDArray[np.float64], splines: int) -> npt.NDArray[np.float64]:
    """Interpola os pontos de Z (waterlines) conforme a densidade especificada."""
    if splines == 0:
        return wl.copy()
    else:
        step = (wl[1] - wl[0]) / splines
        return np.arange(wl[0], wl[-1] + step, step)

def grafico_X_Y(x: npt.NDArray[np.float64], wl: npt.NDArray[np.float64], 
                ct: pd.DataFrame, x_spl: npt.NDArray[np.float64]):
    """Gera gráficos de balizas para cada linha d'água."""
    for k in range(len(wl)):
        # Pega os valores da coluna k+1 (ignorando o cabeçalho)
        y = ct.iloc[1:, k+1].values.astype(float).flatten()
        y_spl = splcubic(x, y, x_spl)
        plot_wlxbal(x_spl, y_spl, wl[k], k == len(wl)-1)

def plot_wlxbal(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64], 
                wl_val: float, is_last: bool):
    """Plota uma curva de baliza para uma linha d'água específica."""
    plt.scatter(x, y, label=f'WL {wl_val:.1f}', s=10)
    plt.plot(x, y, linewidth=1)
    
    if is_last:
        plt.ylim(-1, 17)
        plt.gca().set_aspect(20/8, adjustable='box')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Baliza', fontsize=12)
        plt.ylabel('Meia-boca', fontsize=12)
        plt.title('Curvas de Balizas', fontsize=14)
        plt.legend()
        plt.tight_layout()
        plt.show()

def grafico_Z_Y(x: npt.NDArray[np.float64], wl: npt.NDArray[np.float64], 
                ct: pd.DataFrame, z_spl: npt.NDArray[np.float64]):
    """Gera gráficos de linhas d'água para cada baliza."""
    for k in range(len(x)):
        # Pega os valores da linha k+1 (ignorando o cabeçalho)
        y = ct.iloc[k+1, 1:].values.astype(float).flatten()
        y_spl = splcubic(wl, y, z_spl)
        plot_calxwl(x, z_spl, y_spl, x[k], k == len(x)-1)

def plot_calxwl(x_full: npt.NDArray[np.float64], z: npt.NDArray[np.float64], 
                y: npt.NDArray[np.float64], x_val: float, is_last: bool):
    """Plota uma curva de linha d'água para uma baliza específica."""
    if x_val > (x_full[-1])/2:
        y_plot = -y
        plt.scatter(y_plot, z)
        plt.plot(y_plot, z)
    elif x_val == (x_full[-1])/2:
        plt.scatter(y, z, color='#edca7f')
        plt.plot(y, z, color='#edca7f')
        plt.scatter(-y, z, color='#edca7f')
        plt.plot(-y, z, color='#edca7f')
    else:
        plt.scatter(y, z)
        plt.plot(y, z)
    
    if is_last:
        plt.xlim(-16, 16)
        plt.xlabel('Meia-boca', fontsize=12)
        plt.ylabel('Calado', fontsize=12)
        plt.title('Curvas de Linhas d\'Água', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

def splcubic(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64], 
             x_spl: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Calcula a interpolação por spline cúbica."""
    n = len(x)
    h = np.diff(x)
    a = y.copy()
    
    # Construir matriz A e vetor B
    A = np.zeros((n, n))
    B = np.zeros(n)
    
    # Condições de contorno naturais
    A[0, 0] = 1
    A[-1, -1] = 1
    
    for i in range(1, n-1):
        A[i, i-1] = h[i-1]
        A[i, i] = 2 * (h[i-1] + h[i])
        A[i, i+1] = h[i]
        B[i] = 3 * ((a[i+1] - a[i]) / h[i] - (a[i] - a[i-1]) / h[i-1])
    
    c = np.linalg.solve(A, B)
    b = np.zeros(n-1)
    d = np.zeros(n-1)
    
    for i in range(n-1):
        b[i] = (a[i+1] - a[i]) / h[i] - h[i] * (2 * c[i] + c[i+1]) / 3
        d[i] = (c[i+1] - c[i]) / (3 * h[i])
    
    # Avaliar spline nos pontos x_spl
    y_spl = np.zeros_like(x_spl)
    for i in range(len(x_spl)):
        # Encontrar o intervalo correto
        idx = np.searchsorted(x, x_spl[i]) - 1
        idx = np.clip(idx, 0, n-2)
        
        dx = x_spl[i] - x[idx]
        y_spl[i] = a[idx] + b[idx] * dx + c[idx] * dx**2 + d[idx] * dx**3
        y_spl[i] = max(y_spl[i], 0)  # Não permitir valores negativos
    
    return y_spl

##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################



if __name__ == "__main__":
    main()