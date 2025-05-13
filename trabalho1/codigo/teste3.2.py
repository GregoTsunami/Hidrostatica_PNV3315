# teste 2
# tentativa de correção das props hidrostaticas 2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Union
import numpy.typing as npt

def main():
    # Carrega os dados do arquivo Excel
    data = pd.ExcelFile('super_tanker.xlsx')
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
    
    # Processa os pontos e cálculos hidrostáticos
    pontos = processar_pontos(x, wl, ct, x_spl, z_spl)
    paineis = identificar_paineis(pontos, x_spl, z_spl)
    calcular_hidrostatica(paineis, x_spl, z_spl, pontos)

def spline_x(x: npt.NDArray[np.float64], splines: int) -> npt.NDArray[np.float64]:
    """Interpola os pontos de X (balizas) conforme a densidade especificada."""
    if splines == 0:
        return x.copy()
    else:
        step = (x[1] - x[0]) / (splines + 1)
        return np.arange(x[0], x[-1] + step, step)

def spline_z(wl: npt.NDArray[np.float64], splines: int) -> npt.NDArray[np.float64]:
    """Interpola os pontos de Z (waterlines) conforme a densidade especificada."""
    if splines == 0:
        return wl.copy()
    else:
        step = (wl[1] - wl[0]) / (splines + 1)
        return np.arange(wl[0], wl[-1] + step, step)

def grafico_X_Y(x: npt.NDArray[np.float64], wl: npt.NDArray[np.float64], 
                ct: pd.DataFrame, x_spl: npt.NDArray[np.float64]):
    """Gera gráficos de balizas para cada linha d'água."""
    plt.figure(figsize=(10, 6))
    for k in range(len(wl)):
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
    plt.figure(figsize=(10, 6))
    for k in range(len(x)):
        y = ct.iloc[k+1, 1:].values.astype(float).flatten()
        y_spl = splcubic(wl, y, z_spl)
        plot_calxwl(x, z_spl, y_spl, x[k], k == len(x)-1)

def plot_calxwl(x_full: npt.NDArray[np.float64], z: npt.NDArray[np.float64], 
                y: npt.NDArray[np.float64], x_val: float, is_last: bool):
    """Plota uma curva de linha d'água para uma baliza específica."""
    if x_val > (x_full[-1])/2:
        y_plot = -y
        plt.scatter(y_plot, z, s=10)
        plt.plot(y_plot, z, linewidth=1)
    elif x_val == (x_full[-1])/2:
        plt.scatter(y, z, color='#edca7f', s=10)
        plt.plot(y, z, color='#edca7f', linewidth=1)
        plt.scatter(-y, z, color='#edca7f', s=10)
        plt.plot(-y, z, color='#edca7f', linewidth=1)
    else:
        plt.scatter(y, z, s=10)
        plt.plot(y, z, linewidth=1)
    
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
##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################

def processar_pontos(x, wl, ct, x_spl, z_spl):
    """Processa todos os pontos do casco."""
    pontos = []
    
    # Gera todos os pontos interpolados
    for k in range(len(wl)):
        y = ct.iloc[1:, k+1].values.astype(float).flatten()
        y_spl = splcubic(x, y, x_spl)
        
        for i in range(len(x_spl)):
            pontos.append([x_spl[i], y_spl[i], wl[k]])
    
    # Converte coordenadas
    pontos = conversao_coordenadas(pontos, x_spl, z_spl)
    
    return pontos

def conversao_coordenadas(pontos, x_spl, z_spl):
    """Converte as coordenadas para o sistema do navio."""
    for ponto in pontos:
        ponto[0] -= x_spl[-1]/2  # Centraliza no eixo X
        ponto[2] -= z_spl[-1]    # Ajusta referência Z
    return pontos

def identificar_paineis(pontos, x_spl, z_spl):
    """Identifica os painéis do casco."""
    paineis = []
    n_x = len(x_spl)
    n_z = len(z_spl)
    
    if len(pontos) != n_x * n_z:
        raise ValueError("Número de pontos não corresponde à grade esperada")
    
    for i in range(n_z - 1):
        for j in range(n_x - 1):
            idx = i * n_x + j
            painel = [
                pontos[idx],            # P1: canto inferior esquerdo
                pontos[idx + 1],        # P2: canto inferior direito
                pontos[idx + n_x + 1],  # P3: canto superior direito
                pontos[idx + n_x]       # P4: canto superior esquerdo
            ]
            paineis.append(painel)
    
    return paineis

def areaPainel(painel):
    """Calcula o vetor área de um painel."""
    p1, p2, p3, p4 = np.array(painel)
    v1 = p2 - p1
    v2 = p4 - p1
    v3 = p4 - p3
    v4 = p2 - p3
    area1 = np.cross(v1, v2) / 2
    area2 = np.cross(v4, v3) / 2
    return area1 + area2

def Centros(painel):
    """Calcula o centro geométrico de um painel."""
    return np.mean(painel, axis=0).tolist()

def calcular_hidrostatica(paineis, x_spl, z_spl, pontos):
    """Calcula todas as propriedades hidrostáticas."""
    # Dimensões principais
    Lpp = x_spl[-1] - x_spl[0]  # Comprimento entre perpendiculares
    B = np.max([p[1] for p in pontos]) * 2  # Boca máxima
    T = np.abs(z_spl[-1])  # Calado máximo
    
    # Inicializa variáveis
    total_awl = 0          # Área da linha d'água
    total_volume = 0        # Volume deslocado
    total_momento_x = 0     # Momento longitudinal
    total_momento_y = 0     # Momento transversal
    total_momento_z = 0     # Momento vertical
    total_inercia_x = 0     # Inércia transversal
    total_inercia_y = 0     # Inércia longitudinal
    
    # Pré-calcula centro de flutuação
    lcf, tcf = calcular_centro_flutuacao(paineis)
    
    for painel in paineis:
        area_vec = areaPainel(painel)
        area_mod = np.linalg.norm(area_vec)
        centro = Centros(painel)
        
        # Cálculo do volume
        dz = z_spl[1] - z_spl[0] if len(z_spl) > 1 else 1.0
        volume = area_mod * dz
        total_volume += volume
        
        # Área da linha d'água
        if np.isclose(centro[2], 0, atol=1e-3):
            total_awl += abs(area_vec[2])
            
        # Momentos e inércias
        if centro[2] <= 0:
            total_momento_x += volume * centro[0]
            total_momento_y += volume * centro[1]
            total_momento_z += volume * abs(centro[2])
            
            # Inércia transversal
            y_dist = centro[1] - tcf
            total_inercia_x += (area_mod * dz**3)/12 + area_mod * dz * y_dist**2
            
            # Inércia longitudinal
            x_dist = centro[0] - lcf
            total_inercia_y += (area_mod * dz**3)/12 + area_mod * dz * x_dist**2

    # Cálculo dos centros
    LCB = total_momento_x / total_volume if total_volume != 0 else 0
    TCB = total_momento_y / total_volume if total_volume != 0 else 0
    KB = total_momento_z / total_volume if total_volume != 0 else 0

    # Coeficientes hidrostáticos
    Cb = total_volume / (Lpp * B * T)  # Coeficiente de bloco
    Cp = total_volume / (Lpp * np.trapz([p[1] for p in pontos if np.isclose(p[2], 0, atol=1e-3)], 
                                       x=[p[0] for p in pontos if np.isclose(p[2], 0, atol=1e-3)]))
    
    # Resultados
    print("\nPROPRIEDADES HIDROSTÁTICAS COMPLETAS")
    print("------------------------------------")
    print(f"Volume moldado: {total_volume:.3f} m³")
    print(f"Área da flutuação: {total_awl:.2f} m²")
    print(f"Coeficiente de bloco (Cb): {Cb:.4f}")
    print(f"Momento de inércia transversal (It): {total_inercia_x:.3f} m⁴")
    print(f"Momento de inércia longitudinal (IL): {total_inercia_y:.3f} m⁴")
    print(f"Centro de flutuação (XF): {lcf:.3f} m")
    print(f"Altura metacêntrica transversal (KMt): {KB + total_inercia_x/total_volume:.3f} m")
    print(f"Altura metacêntrica longitudinal (KMl): {KB + total_inercia_y/total_volume:.3f} m")

def calcular_centro_flutuacao(paineis):
    """Calcula LCF (longitudinal) e TCF (transversal) da linha d'água."""
    area_total = 0
    momento_x = 0
    momento_y = 0
    
    for painel in paineis:
        centro = Centros(painel)
        if np.isclose(centro[2], 0, atol=1e-3):  # Apenas linha d'água
            area_vec = areaPainel(painel)
            area = abs(area_vec[2])
            area_total += area
            momento_x += area * centro[0]
            momento_y += area * centro[1]
    
    lcf = momento_x / area_total if area_total != 0 else 0
    tcf = momento_y / area_total if area_total != 0 else 0
    
    return lcf, tcf

if __name__ == "__main__":
    main()